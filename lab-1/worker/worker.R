library(jsonlite)

# --- NATS connection helpers ---

parse_nats_url <- function(url) {
  url <- sub("nats://", "", url)
  parts <- strsplit(url, ":")[[1]]
  list(host = parts[1], port = as.integer(parts[2]))
}

connect_nats <- function(host, port, retries = 15) {
  for (i in seq_len(retries)) {
    con <- tryCatch(
      socketConnection(host, port, open = "r+", blocking = TRUE, timeout = 300),
      error = function(e) NULL
    )
    if (!is.null(con)) {
      cat(sprintf("[worker] Connected to NATS at %s:%d\n", host, port))
      return(con)
    }
    cat(sprintf("[worker] NATS not ready, retry %d/%d...\n", i, retries))
    Sys.sleep(2)
  }
  stop("Cannot connect to NATS after retries")
}

nats_send <- function(con, msg) {
  cat(msg, file = con, sep = "")
}

nats_read_line <- function(con) {
  line <- readLines(con, n = 1, warn = FALSE)
  if (length(line) == 0) return(NA_character_)  # EOF / connection closed
  gsub("\r", "", line)
}

# --- Worker session: subscribe and process until connection drops ---

run_session <- function(con) {
  nats_read_line(con)                                      # INFO {...}
  nats_send(con, "CONNECT {\"verbose\":false}\r\n")
  nats_send(con, "SUB tasks.compute workers 1\r\n")
  nats_send(con, "PING\r\n")
  nats_read_line(con)                                      # +OK or PONG

  cat("[worker] Subscribed to tasks.compute (queue: workers). Ready.\n")

  while (TRUE) {
    line <- tryCatch(nats_read_line(con), error = function(e) NA_character_)

    if (is.na(line)) {
      cat("[worker] Connection lost — will reconnect.\n")
      tryCatch(close(con), error = function(e) NULL)
      return(invisible(NULL))
    }

    if (nchar(line) == 0) next

    if (startsWith(line, "PING")) {
      nats_send(con, "PONG\r\n")
      next
    }

    if (!startsWith(line, "MSG")) next

    # MSG <subject> <sid> [replyTo] <size>
    parts <- strsplit(line, " ")[[1]]

    if (length(parts) == 5) {
      reply_to <- parts[4]
    } else {
      nats_read_line(con)
      next
    }

    payload <- nats_read_line(con)
    if (is.na(payload)) {
      cat("[worker] Connection lost reading payload — will reconnect.\n")
      tryCatch(close(con), error = function(e) NULL)
      return(invisible(NULL))
    }

    task <- fromJSON(payload)
    op   <- if (!is.null(task$type)) task$type else "matmul"

    if (op == "matmul") {
      val    <- sum(task$row_a * task$col_b)
      # digits = NA → full float64 precision (avoids ~5e-5 error from default 4 d.p.)
      result <- toJSON(list(i = task$i, j = task$j, val = val), auto_unbox = TRUE, digits = NA)

    } else if (op == "transpose") {
      result <- toJSON(list(row_idx = task$row_idx, row = task$row), auto_unbox = TRUE, digits = NA)

    } else if (op == "solve") {
      # fromJSON already gives us an R matrix for a 2-D JSON array — use directly.
      x      <- solve(task$A, task$b)
      result <- toJSON(list(x = as.vector(x)), auto_unbox = TRUE, digits = NA)

    } else if (op == "ata_row") {
      # One row of A: partial contributions to A'A and A'Y.
      # Coordinator accumulates these over all M rows without ever storing A.
      row_a  <- task$row_a
      y      <- task$y
      ata_p  <- outer(row_a, row_a)   # K×K partial A'A contribution
      aty_p  <- row_a * y             # K×1 partial A'Y contribution
      result <- toJSON(
        list(ata_partial = as.vector(ata_p), aty_partial = aty_p),
        auto_unbox = TRUE, digits = NA
      )

    } else {
      cat(sprintf("[worker] Unknown task type: %s\n", op))
      next
    }

    resp_size <- nchar(result, type = "bytes")
    nats_send(con, sprintf("PUB %s %d\r\n%s\r\n", reply_to, resp_size, result))
  }
}

# --- Main: reconnect loop ---

addr <- parse_nats_url(Sys.getenv("NATS_URL", "nats://nats:4222"))

repeat {
  tryCatch({
    con <- connect_nats(addr$host, addr$port)
    run_session(con)
  }, error = function(e) {
    cat(sprintf("[worker] Error: %s\n", e$message))
  })
  cat("[worker] Reconnecting in 3s...\n")
  Sys.sleep(3)
}
