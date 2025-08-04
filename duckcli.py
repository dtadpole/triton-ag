#!/usr/bin/env python3
# duckfs.py — DuckDB file-system shell (always in-memory)
# Queries local files via SQL; no DB path needed.

import os, sys, time, shlex, glob
import duckdb

# --- Interactive input handling ---------------------------------------------
def _attach_tty_if_needed():
    """
    Ensure we read from an interactive terminal. If stdin is not a TTY (e.g.,
    launched from an IDE or with redirection), switch to /dev/tty on POSIX.
    """
    if sys.stdin is not None and sys.stdin.isatty():
        return  # already interactive
    if os.name == "posix":
        try:
            sys.stdin = open("/dev/tty", "r", encoding="utf-8", errors="replace")
            return
        except Exception:
            pass
    # If we get here, we'll try to proceed; input() may still raise EOFError.

_attach_tty_if_needed()

# Optional line editing/history (POSIX)
try:
    import readline  # type: ignore
    HISTFILE = os.path.expanduser("~/.duckfs_history")
    try:
        readline.read_history_file(HISTFILE)
    except FileNotFoundError:
        pass
except Exception:
    readline = None
    HISTFILE = None

TIMER = True  # .timer off to disable timings
con = duckdb.connect()  # always in-memory

def _print_table(cols, rows, max_width=100):
    if not rows:
        print("(0 rows)"); return
    widths = [len(str(c)) for c in cols]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = min(max(widths[i], len("" if v is None else str(v))), max_width)
    def fmt(row):
        return " | ".join(str("" if v is None else v)[:widths[i]].ljust(widths[i]) for i, v in enumerate(row))
    print(fmt(cols))
    print("-+-".join("-"*w for w in widths))
    for r in rows:
        print(fmt(r))

def exec_and_print(sql: str):
    t0 = time.time()
    try:
        rel = con.sql(sql)            # relation, portable across versions
        cols = list(rel.columns)      # column names
        rows = rel.fetchall()         # list of tuples
        _print_table(cols, rows)
        print(f"{len(rows)} row(s) in {(time.time()-t0)*1000:.1f} ms")
    except Exception as e:
        print(f"error: {e}")

def stmt_complete(buffer: str) -> bool:
    """Complete when buffer ends with ';' and we're not inside single quotes."""
    s = buffer.strip()
    if not s:
        return False
    in_single, esc = False, False
    for ch in s:
        if ch == "'" and not esc:
            in_single = not in_single
        esc = (ch == "\\" and not esc)
    return (not in_single) and s.endswith(";")

def do_ls(pattern: str | None):
    pat = pattern or "*"
    items = sorted(glob.glob(pat))
    if not items:
        print("(no matches)"); return
    for x in items:
        print(x)

def handle_meta(line: str):
    global TIMER, con
    parts = shlex.split(line)
    cmd = parts[0][1:] if parts else ""
    args = parts[1:]

    if cmd in ("q","quit","exit"):
        if HISTFILE and readline:
            try: readline.write_history_file(HISTFILE)
            except Exception: pass
        sys.exit(0)

    if cmd == "help":
        print("""\
Meta commands:
  .pwd                 show current directory
  .cd DIR              change directory
  .ls [PATTERN]        list files (glob)
  .read FILE.sql       execute SQL from a file
  .tables              list base tables
  .schema [TABLE]      show schema (all or one table)
  .timer on|off        toggle elapsed-time output
  .install EXT [...]   INSTALL and LOAD DuckDB extensions (e.g., httpfs, json)
  .load EXT [...]      LOAD already-installed extensions
  .help                this help
  .quit                exit
""")
        return

    if cmd == "pwd":
        print(os.getcwd()); return

    if cmd == "cd":
        if not args: print("usage: .cd DIR"); return
        try: os.chdir(args[0]); print(os.getcwd())
        except Exception as e: print(f"error: {e}")
        return

    if cmd == "ls":
        do_ls(args[0] if args else None); return

    if cmd == "read":
        if not args: print("usage: .read FILE.sql"); return
        try:
            with open(args[0], "r", encoding="utf-8") as f:
                script = f.read()
            con.execute(script)
            print("OK")
        except Exception as e:
            print(f"error: {e}")
        return

    if cmd == "tables":
        exec_and_print("""
        SELECT table_schema AS schema, table_name AS name
        FROM information_schema.tables
        WHERE table_type='BASE TABLE'
        ORDER BY 1,2;
        """); return

    if cmd == "schema":
        if args:
            exec_and_print(f"PRAGMA table_info('{args[0]}');")
        else:
            exec_and_print("""
            SELECT table_schema AS schema, table_name AS name
            FROM information_schema.tables
            WHERE table_type='BASE TABLE'
            ORDER BY 1,2;
            """)
        return

    if cmd == "timer":
        if not args or args[0] not in ("on","off"):
            print("usage: .timer on|off"); return
        TIMER = (args[0] == "on"); print(f"timer {'on' if TIMER else 'off'}")
        return

    if cmd == "install":
        if not args: print("usage: .install EXT [...]"); return
        for ext in args:
            try:
                con.install_extension(ext); con.load_extension(ext)
                print(f"installed+loaded: {ext}")
            except Exception as e:
                print(f"error: {ext}: {e}")
        return

    if cmd == "load":
        if not args: print("usage: .load EXT [...]"); return
        for ext in args:
            try:
                con.load_extension(ext); print(f"loaded: {ext}")
            except Exception as e:
                print(f"error: {ext}: {e}")
        return

    print(f"unknown: .{cmd} (try .help)")

def main():
    print("DuckDB FS shell — in-memory DB. Query local files directly.")
    print("Use .help for commands. End SQL with ';'. Ctrl+C clears the buffer.")
    print(f"cwd: {os.getcwd()}")
    buffer = ""
    while True:
        try:
            prompt = "duckdb> " if not buffer else "   ...> "
            # Use input(); if stdin isn't interactive, _attach_tty_if_needed() tried to fix it.
            line = input(prompt)
        except EOFError:
            # If there's truly no interactive input, don't just disappear—tell the user.
            print("\n(no interactive input detected; are you running without a TTY?)")
            print("Try running from a real terminal/SSH session.")
            break
        except KeyboardInterrupt:
            print(); buffer = ""; continue

        if not buffer and line.strip().startswith("."):
            handle_meta(line.strip()); continue

        buffer += (line + "\n")
        if stmt_complete(buffer):
            sql = buffer.strip()[:-1]  # drop trailing ';'
            exec_and_print(sql)
            buffer = ""

    if HISTFILE and readline:
        try: readline.write_history_file(HISTFILE)
        except Exception: pass

if __name__ == "__main__":
    main()
