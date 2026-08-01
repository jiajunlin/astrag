(* Parsing helpers. *)

let rec skip_spaces s i =
  if i < String.length s && s.[i] = ' ' then skip_spaces s (i + 1) else i

let parse_int s =
  int_of_string (String.trim s)
