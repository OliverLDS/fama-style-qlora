#!/usr/bin/env Rscript
# Provider errors are classified without printing credential-bearing URLs.
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=4L)stop('Expected run, phase, job index and output path.',call.=FALSE)
suppressPackageStartupMessages(library(jsonlite))
root<-args[[1]];phase<-args[[2]];index<-as.integer(args[[3]]);destination<-args[[4]]
config<-fromJSON(file.path(root,'config.json'),simplifyVector=FALSE)
if(!identical(config$model,'gemini-3-flash-preview')||!isTRUE(config$free_tier_only)||
   !isTRUE(config$free_project_confirmed_by_user)||isTRUE(config$paid_fallback)||file.exists(destination))stop('Free-only contract failed.',call.=FALSE)
if(!phase%in%c('generation','verification'))stop('Invalid phase.',call.=FALSE)
jobs<-lapply(readLines(file.path(root,phase,'jobs.private.jsonl'),warn=FALSE),function(l)fromJSON(l,simplifyVector=FALSE))
if(is.na(index)||index<1L||index>length(jobs))stop('Invalid index.',call.=FALSE)
job<-jobs[[index]];started<-Sys.time()
result<-list(status='request_failed',prompt_sha256=job$prompt_sha256,model_requested=config$model,
             phase=phase,job_index=index,free_project_user_confirmed=TRUE,paid_fallback=FALSE,
             thinking_level=config$thinking_level,client='inferencer native HTTP helper')
response<-tryCatch(withCallingHandlers({
 # Public query_gemini does not expose thinkingConfig; use the same package's HTTP helper.
 if(!config$thinking_level%in%c('low','medium'))stop('Bounded reasoning required.',call.=FALSE)
 response<-inferencer:::.perform_json_request(
   url=paste0('https://generativelanguage.googleapis.com/v1beta/models/',config$model,':generateContent'),
   headers=list('Content-Type'='application/json','x-goog-api-key'=Sys.getenv('GEMINI_API_KEY')),
   body=list(contents=list(list(role='user',parts=list(list(text=job$prompt)))),
     generationConfig=list(temperature=config$temperature,topP=config$top_p,topK=config$top_k,
       maxOutputTokens=config$max_output_tokens,responseMimeType='application/json',
       thinkingConfig=list(thinkingLevel=config$thinking_level))),timeout=600)
 status<-httr2::resp_status(response)
 if(status>=400L)stop(paste('HTTP',status),call.=FALSE)
 httr2::resp_body_json(response,simplifyVector=FALSE)
 },warning=function(w)invokeRestart('muffleWarning'),message=function(m)invokeRestart('muffleMessage')),
 error=function(e){
   msg<-conditionMessage(e)
   result$error_category<<-if(grepl('429|RESOURCE_EXHAUSTED|quota',msg,ignore.case=TRUE))'free_quota_or_rate_limit' else 'provider_error_suppressed'
   NULL
 })
if(!is.null(response)){
 result$raw_response<-response;result$model_version<-response$modelVersion;result$usage<-response$usageMetadata
 if(length(response$candidates)==1L){
  candidate<-response$candidates[[1]];parts<-Filter(function(p)!isTRUE(p$thought)&&is.character(p$text),candidate$content$parts)
  result$text<-paste(vapply(parts,function(p)p$text,character(1)),collapse='');result$finish_reason<-candidate$finishReason
  result$status<-if(identical(candidate$finishReason,'STOP')&&nzchar(trimws(result$text)))'generated' else 'incomplete_or_blocked'
 }else result$status<-'blocked_or_ambiguous_candidates'
}
result$elapsed_seconds<-as.numeric(difftime(Sys.time(),started,units='secs'))
result$inferencer_version<-as.character(utils::packageVersion('inferencer'))
result$recorded_at_utc<-format(Sys.time(),tz='UTC',usetz=TRUE)
write_json(result,destination,auto_unbox=TRUE,null='null',pretty=TRUE)
quit(status=if(identical(result$status,'generated'))0L else 1L)
