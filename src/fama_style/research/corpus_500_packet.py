"""Freeze a mixed-teacher 500-pair pilot and matched held-out input variants."""
import json
from pathlib import Path

from fama_style.research.corpus_200_packet import file_sha, read_rows, verify_parent, write_rows
from fama_style.training import corpus_500_pilot as runtime


def build(training, parent_packet, gemini_inputs, output):
    if output.exists():
        raise ValueError('refuse_overwrite_frozen_pilot_packet')
    verify_parent(training)
    parent_manifest = json.loads((parent_packet / 'manifest.json').read_text())
    for name, digest in parent_manifest['files'].items():
        if file_sha(parent_packet / name) != digest:
            raise ValueError('parent_200_packet_changed')
    teacher_manifest = json.loads((gemini_inputs.parent / 'accepted_manifest.json').read_text())
    for name, digest in teacher_manifest['files'].items():
        if file_sha(gemini_inputs.parent / name) != digest:
            raise ValueError('held_out_teacher_inputs_changed')
    train = read_rows(training)
    if len(train) != 500 or any(r['split'] != 'training' or not r['training_ready'] or
        r['quality']['text_pair_status'] != 'automated_pilot_approved' for r in train):
        raise ValueError('approved_500_training_cohort_required')
    old_config = json.loads((parent_packet / 'upload/corpus_200_config.json').read_text())
    old_rows = read_rows(parent_packet / 'upload/corpus_200_pairs.jsonl')
    old_dev = [r for r in old_rows if r['split'] == 'development']
    if [r['example_id'] for r in old_dev] != old_config['development_ids']:
        raise ValueError('parent_development_order_changed')
    old_train = [r for r in old_rows if r['split'] == 'training']
    if len(old_train) != 200:
        raise ValueError('parent_200_training_scope')
    for a, b in zip(train[:200], old_train):
        if (a['id'], a['source_sha256'], a['target_sha256']) != (b['example_id'], b['source_sha256'], b['target_sha256']):
            raise ValueError('original_200_text_or_order_changed')
    gemini = read_rows(gemini_inputs)
    if [r['content_id'] for r in gemini] != old_config['detector_case_ids']:
        raise ValueError('frozen_eight_Gemini_cases_required')
    by = {r['example_id']: r for r in old_dev}
    export = []
    for row in train:
        export.append(dict(example_id=row['id'], content_id=row['id'], parent_id=row['parent_id'],
            paper_id=row['paper_id'], style_group=row['style_group'], authors=row['authors'], split='training',
            source=row['source'], target=row['target'], source_sha256=row['source_sha256'], target_sha256=row['target_sha256'],
            prompt=runtime.prompt(row['source'], training=True), pilot_eligible=True, input_condition='training_mixed_teacher'))
    for row in old_dev:
        export.append(dict(row, content_id=row['example_id'], input_condition='existing_input', prompt=runtime.prompt(row['source'])))
    for row in gemini:
        if row['acceptance'] != 'automated_meaning_review' or row['model'] != 'gemini-3-flash-preview':
            raise ValueError('unreviewed_Gemini_input')
        parent = by[row['content_id']]
        if row['reference_sha256'] != parent['target_sha256'] or row['source_sha256'] != runtime.sha(row['source']):
            raise ValueError('Gemini_reference_or_source_binding')
        export.append(dict(parent, example_id=row['content_id'] + '::gemini_input_v1', content_id=row['content_id'],
            input_condition='gemini_input', source=row['source'], source_sha256=row['source_sha256'], prompt=runtime.prompt(row['source'])))
    output.mkdir(parents=True)
    (output / 'upload').mkdir()
    data = output / 'upload/corpus_500_pairs.jsonl'
    write_rows(data, export)
    gemini_ids = [r['example_id'] for r in export if r['input_condition'] == 'gemini_input']
    # Freeze two initial scan cases before any detector outcome. Expand to the
    # remaining six without selecting cases by favorable detector scores.
    priority = [next(r['example_id'] for r in export if r['input_condition'] == 'gemini_input' and r['style_group'] == g)
                for g in runtime.GROUPS[:2]]
    config = dict(protocol=runtime.PROTOCOL, model_id=runtime.MODEL_ID, model_revision=runtime.REVISION,
        hyperparameters=runtime.HYPER, decoding=runtime.DECODING, packages=runtime.PACKAGES,
        max_runtime_seconds=runtime.MAX_SECONDS, training_authorized=True,
        quantization='NF4 double quantization; FP32 compute',
        runtime_source_sha256=file_sha(Path(runtime.__file__)), dataset_sha256=file_sha(data),
        training_ids=[r['example_id'] for r in export if r['split'] == 'training'],
        development_ids=[r['example_id'] for r in export if r['split'] == 'development'],
        training_probe_ids=old_config['training_probe_ids'],
        gemini_development_ids=gemini_ids, parent_detector_content_ids=old_config['detector_case_ids'],
        detector_case_ids=gemini_ids, detector_priority_ids=priority,
        training_instruction=runtime.TRAIN_INSTRUCTION, inference_instruction=runtime.INSTRUCTION,
        training_cohort_sha256=file_sha(training), parent_200_packet_manifest_sha256=file_sha(parent_packet / 'manifest.json'),
        Gemini_accepted_manifest_sha256=file_sha(gemini_inputs.parent / 'accepted_manifest.json'),
        authorization_scope=dict(mode='prepared_for_manual_Kaggle_UI_execution', epochs=1, updates=250,
            training_pairs=500, development_contents=16, development_input_variants=24, matched_generations=48,
            GPU='one Kaggle T4', max_seconds=5400, automatic_submission=False, automatic_retries=False, full_training=False),
        evaluation_limits='Mixed teachers; previously exposed paper-held-out development within known author groups. Same references for eight Gemini input variants. No independent gold or Fama-style/detector success claimed.')
    path = output / 'upload/corpus_500_config.json'
    runtime.write_json(path, config)
    runtime.load_inputs(data, path)
    plan = dict(protocol=runtime.PROTOCOL, status='prepared_not_executed', training_pairs=500,
        training_groups={g: 125 for g in runtime.GROUPS}, training_papers=len({r['paper_id'] for r in train}),
        development_contents=16, development_papers=8, development_input_conditions={'existing_input':16,'gemini_input':8},
        epochs=1, optimizer_steps=250, matched_generations=48, max_runtime_seconds=5400,
        model_id=runtime.MODEL_ID, model_revision=runtime.REVISION, training_instruction=runtime.TRAIN_INSTRUCTION,
        inference_instruction=runtime.INSTRUCTION, base_initialization='Fresh pinned base; no prior adapter continuation',
        detector_initial_cases=2, detector_initial_texts=6, detector_full_cases=8, detector_full_texts=24,
        detector_priority_ids=priority, GPU_jobs_submitted=0, root_1b_pipeline_modified=False,
        sealed_test_examples_read=0, separate_teacher_tracks='Deferred until mixed-500 results are reviewed')
    runtime.write_json(output / 'plan.json', plan)
    runtime.write_json(output / 'manifest.json', dict(protocol=runtime.PROTOCOL,
        runtime_source_sha256=config['runtime_source_sha256'], builder_source_sha256=file_sha(Path(__file__)),
        files={str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return plan
