#!/usr/bin/env Rscript
# Single request adapter. API errors are never printed because their URLs may contain a key.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3L) stop('Expected packet directory, row index, response path.', call.=FALSE)
suppressPackageStartupMessages(library(jsonlite))
packet <- args[[1]]; index <- as.integer(args[[2]]); destination <- args[[3]]
config <- fromJSON(file.path(packet, 'config.json'), simplifyVector=FALSE)
if (!identical(config$model, 'gemini-3-flash-preview') ||
    !isTRUE(config$free_tier_only) || !isTRUE(config$free_project_confirmed_by_user) ||
    isTRUE(config$paid_fallback) || file.exists(destination)) stop('Free-only trial contract failed.', call.=FALSE)
items <- lapply(readLines(file.path(packet, 'inputs.private.jsonl'), warn=FALSE),
                function(line) fromJSON(line, simplifyVector=FALSE))
if (is.na(index) || index < 1L || index > 8L || length(items) != 8L) stop('Invalid request index.', call.=FALSE)
row <- items[[index]]
started <- Sys.time()
result <- list(example_id=row$example_id, prompt_sha256=row$prompt_sha256,
               model_requested=config$model, status='request_failed', text=NULL,
               free_project_user_confirmed=TRUE, paid_fallback=FALSE)
response <- tryCatch(withCallingHandlers({
    inferencer::query_gemini(row$prompt, model=config$model,
      url0='https://generativelanguage.googleapis.com/v1beta/models',
      temperature=config$temperature, top_p=config$top_p, top_k=config$top_k,
      max_tokens=config$max_output_tokens, json_list=TRUE, timeout=150)
  }, warning=function(w) invokeRestart('muffleWarning'), message=function(m) invokeRestart('muffleMessage')),
  error=function(e) NULL)
if (!is.null(response)) {
  result$raw_response <- response
  result$model_version <- response$modelVersion
  result$usage <- response$usageMetadata
  if (length(response$candidates) == 1L) {
    candidate <- response$candidates[[1]]
    result$finish_reason <- candidate$finishReason
    parts <- candidate$content$parts
    final <- Filter(function(part) !isTRUE(part$thought) && is.character(part$text), parts)
    result$text <- paste(vapply(final, function(part) part$text, character(1)), collapse='')
    result$status <- if (identical(candidate$finishReason, 'STOP') && nzchar(trimws(result$text)))
                        'generated' else 'incomplete_or_blocked'
  } else result$status <- 'blocked_or_ambiguous_candidates'
}
result$elapsed_seconds <- as.numeric(difftime(Sys.time(), started, units='secs'))
result$inferencer_version <- as.character(utils::packageVersion('inferencer'))
result$recorded_at_utc <- format(Sys.time(), tz='UTC', usetz=TRUE)
write_json(result, destination, auto_unbox=TRUE, null='null', pretty=TRUE)
quit(status=if (identical(result$status, 'generated')) 0L else 1L)
