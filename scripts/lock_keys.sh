
# Locks key files by making them read-only

cd "$HOME/PycharmProjects/facial-recognition/" || echo "Permission denied: key files not found" && exit
chmod 0400 _keys # directory is owner-read-only
chmod -R 0400 _keys # key files are owner-read-only\
# chmod tables

: '

General:
+-----------------------------------------------------------+
| Octal | Decimal |       Permission       | Representation |
|-------|---------|------------------------|----------------|
|  000  |    0    | none                   |      ---       |
|  010  |    2    | write                  |      -w-       |
|  001  |    1    | execute                |      --x       |
|  011  |    3    | write + execute        |      -wx       |
|  100  |    4    | read                   |      r--       |
|  101  |    5    | read + execute         |      r-x       |
|  110  |    6    | read + write           |      rw-       |
|  111  |    7    | read + write + execute |      rwx       |
+-----------------------------------------------------------+

Specific:
+----------------------------+
| Octal |     Permission     |
|-------|--------------------|
|  400  | read by owner      |
|  040  | read by group      |
|  004  | read by anybody    |
|  200  | write by owner     |
|  020  | write by group     |
|  002  | write by anybody   |
|  100  | execute by owner   |
|  010  | execute by group   |
|  001  | execute by anybody |
+----------------------------+

'