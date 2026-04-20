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
  if (length(line) == 0) return("")
  gsub("\r", "", line)
}

# --- Main ---

addr <- parse_nats_url(Sys.getenv("NATS_URL", "nats://nats:4222"))
con  <- connect_nats(addr$host, addr$port)

# NATS handshake
nats_read_line(con)                                      # INFO {...}
nats_send(con, "CONNECT {\"verbose\":false}\r\n")
nats_send(con, "SUB tasks.compute workers 1\r\n")
nats_send(con, "PING\r\n")
nats_read_line(con)                                      # +OK or PONG

cat("[worker] Subscribed to tasks.compute (queue: workers). Ready.\n")

# Message loop
while (TRUE) {
  line <- tryCatch(nats_read_line(con), error = function(e) "")

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
    # No reply-to — read payload and discard
    nats_read_line(con)
    next
  }

  payload <- nats_read_line(con)
  task    <- fromJSON(payload)

  # Dot product: C[i,j] = A[i,:] . B[:,j]
  val    <- sum(task$row_a * task$col_b)
  # digits = NA → full float64 precision (avoids ~5e-5 error from default 4 d.p.)
  result <- toJSON(list(i = task$i, j = task$j, val = val), auto_unbox = TRUE, digits = NA)

  resp_size <- nchar(result, type = "bytes")
  nats_send(con, sprintf("PUB %s %d\r\n%s\r\n", reply_to, resp_size, result))
}
