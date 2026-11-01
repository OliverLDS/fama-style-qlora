"""Derive an isolated DPO counterpart from the exact completed SFT runtime."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SFT_SHA='08b399aa43ccf35b07148284f0d0cc0f67a5f9624915f29aa3ecf2fd9ac023ca'


def derive(source):
    if hashlib.sha256(source.encode()).hexdigest()!=SFT_SHA:raise ValueError('completed_SFT_runtime_changed')
    def replace(old,new):
        nonlocal source
        if source.count(old)!=1:raise ValueError('DPO_derivation_anchor_changed')
        source=source.replace(old,new)
    replace('One bounded stronger SFT continuation; frozen prior artifacts remain unchanged.',
            'Matched bounded DPO continuation; fixed starting-adapter reference and beta 0.1.')
    replace("PROTOCOL='sft_strength_pilot_v1'","PROTOCOL='dpo_strength_pilot_v1'")
    replace("HYPER={k:v for k,v in prior.HYPER.items() if k!='beta'}","HYPER=dict(prior.HYPER)")
    replace("if set(config['initial_chosen_scores'])!=set(by_id):raise ValueError('seventeen_frozen_initial_scores_required')",'''if set(config['initial_chosen_scores'])!=set(by_id) or set(config['reference_scores'])!=set(by_id):
        raise ValueError('seventeen_frozen_reference_pairs_required')
    for pid,row in by_id.items():
        for kind in ('chosen','rejected'):
            score=config['reference_scores'][pid][kind];n=prior.validate_sequence(row['encoded'][kind])[1]
            if (score['supervised_tokens']!=n or not math.isfinite(score['total_log_probability'])
                    or score['total_log_probability']>0 or not math.isclose(score['mean_nll'],-score['total_log_probability']/n)):
                raise ValueError('invalid_frozen_reference_score')
        if config['initial_chosen_scores'][pid]!=config['reference_scores'][pid]['chosen']:
            raise ValueError('chosen_reference_score_mismatch')''')
    replace("[(r,r['chosen'],r['encoded']['chosen'],r['pair_id']) for r in pairs]",
            "[(r,r[kind],r['encoded'][kind],r['pair_id']+'__'+kind) for r in pairs for kind in ('chosen','rejected')]")
    replace("for seq in (pairs[0]['encoded']['chosen'],development[0]['encoded']):",
            "for seq in (pairs[0]['encoded']['chosen'],pairs[0]['encoded']['rejected'],development[0]['encoded']):")
    replace("        report['control_exact_matches']=8\n",'''        report['control_exact_matches']=8
        phase('verify_fixed_reference');reference={}
        initial_chosen={r['pair_id']:{k:r[k] for k in ('total_log_probability','mean_nll','supervised_tokens')}
            for r in report['initial_training_scores']}
        for row in pairs:
            pid=row['pair_id'];reference[pid]={'chosen':initial_chosen[pid],
                'rejected':score(row['encoded']['rejected'],'initial_rejected_reference')}
            for kind in ('chosen','rejected'):
                measured=reference[pid][kind];gold=config['reference_scores'][pid][kind]
                if measured['supervised_tokens']!=gold['supervised_tokens'] or abs(measured['total_log_probability']-gold['total_log_probability'])>1e-3:
                    raise RuntimeError('initial_reference_likelihood_changed')
        if fingerprint(get_peft_model_state_dict(model))!=initial:raise RuntimeError('reference_scoring_changed_adapter')
        cache_hash=prior.sha(json.dumps(reference,sort_keys=True))
        report['reference_cache_sha256']=cache_hash
        prior.write_json(output/'reference_scores.json',{'scores':reference,'cache_sha256':cache_hash,
            'initial_adapter_tensor_sha256':initial,'source_adapter_sha256':config['adapter_sha256'],
            'dataset_sha256':config['dataset_sha256'],'model_id':config['model_id'],'model_revision':config['model_revision'],
            'reference_policy':'unchanged_original_expanded_epoch_2'})
''')
    replace("phase('train_sft');torch.manual_seed(1729);model.train();optimizer=", "phase('initial_policy_reference_identity');torch.manual_seed(1729);model.train();optimizer=")
    replace("        report['optimizer_initially_empty']=not bool(optimizer.state);by_id={r['pair_id']:r for r in pairs}\n",'''        report['optimizer_initially_empty']=not bool(optimizer.state);by_id={r['pair_id']:r for r in pairs}
        if optimizer.state:raise RuntimeError('fresh_optimizer_required')
        identity=[]
        for row in pairs:
            pid=row['pair_id'];current={k:score(row['encoded'][k],'initial_policy_identity') for k in ('chosen','rejected')}
            pc,pr=(current[k]['total_log_probability'] for k in ('chosen','rejected'))
            rc,rr=(reference[pid][k]['total_log_probability'] for k in ('chosen','rejected'))
            delta=max(abs(pc-rc),abs(pr-rr));loss=prior.tokens.dpo_loss(pc,pr,rc,rr,HYPER['beta'])
            if delta>1e-3 or abs(loss-math.log(2))>2e-4:raise RuntimeError('initial_policy_reference_identity_failed')
            identity.append({'pair_id':pid,'max_logp_difference':delta,'dpo_loss':loss})
        report['initial_policy_reference_identity']=identity;phase('train_dpo')
''')
    replace('''                    value,n=total_logp(by_id[pid]['encoded']['chosen'],'sft_backward');loss=-value/n/len(group['pair_ids'])
                    losses.append(float((-value/n).detach().item()));loss.backward();del value,loss''', '''                    row=by_id[pid]
                    policy={k:score(row['encoded'][k],'dpo_policy_nograd') for k in ('chosen','rejected')}
                    pc,pr=(policy[k]['total_log_probability'] for k in ('chosen','rejected'))
                    rc,rr=(reference[pid][k]['total_log_probability'] for k in ('chosen','rejected'))
                    coefficient=prior.preference_coefficients(pc,pr,rc,rr,HYPER['beta']);losses.append(coefficient['loss'])
                    for kind in ('chosen','rejected'):
                        value,_=total_logp(row['encoded'][kind],'dpo_backward')
                        if not math.isclose(float(value.detach().item()),policy[kind]['total_log_probability'],rel_tol=1e-6,abs_tol=1e-3):
                            raise RuntimeError('two_pass_policy_not_deterministic')
                        (value*(coefficient[kind+'_gradient']/len(group['pair_ids']))).backward();del value''')
    replace("torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.,error_if_nonfinite=True);optimizer.step();state['optimizer_steps']=step",'''torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.,error_if_nonfinite=True)
                if prior.sha(json.dumps(reference,sort_keys=True))!=cache_hash:raise RuntimeError('reference_cache_mutated')
                optimizer.step();state['optimizer_steps']=step''')
    replace("'learning_rate':lr,'loss':sum(losses)/len(losses)",
            "'learning_rate':lr,'loss':sum(losses)/len(losses),'reference_cache_sha256':cache_hash")
    replace("        if state['optimizer_steps']!=22 or len(report['final_generations'])!=19 or counts['generation_requests']!=27:",'''        report['final_preference_scores']=[]
        chosen_by={r['pair_id']:{k:r[k] for k in ('total_log_probability','mean_nll','supervised_tokens')} for r in report['final_training_scores']}
        for row in pairs:
            report['final_preference_scores'].append({'pair_id':row['pair_id'],'chosen':chosen_by[row['pair_id']],
                'rejected':score(row['encoded']['rejected'],'final_rejected_score')})
        if prior.sha(json.dumps(reference,sort_keys=True))!=cache_hash:raise RuntimeError('reference_cache_mutated')
        if state['optimizer_steps']!=22 or len(report['final_generations'])!=19 or counts['generation_requests']!=27:''')
    # Output/progress wording is changed only in this new module.
    source=source.replace('Stronger SFT','Matched DPO').replace('sft_strength_v1','dpo_strength_v1')
    source=source.replace('22 SFT updates at 1e-5','22 DPO updates at 1e-5 / beta 0.1')
    source=source.replace('explicit_bounded_sft_authorization_required','explicit_bounded_dpo_authorization_required')
    compile(source,'dpo_strength_pilot.py','exec')
    return source


def build(output=None):
    output=output or ROOT/'src/fama_style/training/dpo_strength_pilot.py'
    source=derive((ROOT/'src/fama_style/training/sft_strength_pilot.py').read_text())
    if output.exists() and output.read_text()!=source:raise ValueError('preserve_existing_DPO_runtime')
    output.write_text(source);return output


if __name__=='__main__':print(build())
