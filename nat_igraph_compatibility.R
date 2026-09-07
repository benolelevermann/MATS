# Compatibility patch for legacy nat code with igraph >= 2.
#
# nat versions used by the scEvoView workflow still call the removed aliases
# graph.dfs()/graph.bfs() and their old argument names. This source file adapts
# those calls in the loaded nat namespace. It does not modify installed package
# files and must be sourced once per fresh R session.

patch_nat_igraph_compatibility <- function() {
  if (!requireNamespace("igraph", quietly = TRUE)) {
    stop(
      "igraph is not installed. Run: install.packages('igraph', repos = 'https://cloud.r-project.org')"
    )
  }
  if (!requireNamespace("nat", quietly = TRUE)) {
    stop(
      "nat is not installed. Run: install.packages('nat', repos = c('https://natverse.r-universe.dev', 'https://cloud.r-project.org'))"
    )
  }

  nat_namespace <- asNamespace("nat")
  patched_functions <- character()

  for (function_name in ls(nat_namespace, all.names = TRUE)) {
    function_object <- get(
      function_name,
      envir = nat_namespace,
      inherits = FALSE
    )
    if (!is.function(function_object)) next

    original_body <- paste(
      deparse(body(function_object), width.cutoff = 500),
      collapse = "\n"
    )
    patched_body <- gsub(
      "igraph::graph\\.dfs([[:space:]]*\\()",
      "igraph::dfs\\1",
      original_body
    )
    patched_body <- gsub(
      "graph\\.dfs([[:space:]]*\\()",
      "igraph::dfs\\1",
      patched_body
    )
    patched_body <- gsub(
      "igraph::graph\\.bfs([[:space:]]*\\()",
      "igraph::bfs\\1",
      patched_body
    )
    patched_body <- gsub(
      "graph\\.bfs([[:space:]]*\\()",
      "igraph::bfs\\1",
      patched_body
    )
    patched_body <- gsub(
      "neimode([[:space:]]*)=",
      "mode\\1=",
      patched_body
    )
    patched_body <- gsub(
      "father([[:space:]]*)=",
      "parent\\1=",
      patched_body
    )

    if (identical(original_body, patched_body)) next

    body(function_object) <- parse(
      text = patched_body,
      keep.source = FALSE
    )[[1]]

    binding_was_locked <- bindingIsLocked(function_name, nat_namespace)
    if (binding_was_locked) unlockBinding(function_name, nat_namespace)
    assign(function_name, function_object, envir = nat_namespace)
    if (binding_was_locked) lockBinding(function_name, nat_namespace)

    patched_functions <- c(patched_functions, function_name)
  }

  message(
    "nat/igraph compatibility patch: ",
    length(patched_functions),
    " function(s) patched",
    if (length(patched_functions)) {
      paste0(" (", paste(patched_functions, collapse = ", "), ")")
    } else {
      ""
    }
  )
  invisible(patched_functions)
}

patch_nat_igraph_compatibility()
