import sys

def all_shapes(max_dim):
    def all_shapes_inner(max_dim):
        if max_dim == 0:
            return [""]

        all_shapes_prev = all_shapes_inner(max_dim - 1)
        return [e + sh for e in "NU" for sh in all_shapes_prev]

    #def strip_leading_n(s):
    #    while s.endswith("N"):
    #        s = s[:-1]
    #    return s

    def strip_leading_n(s):
        return s

    return [strip_leading_n(s) for s in all_shapes_inner(max_dim)]

def pad(*s):
    l = max([len(e) for e in s])
    return [e + "N" * (l - len(e)) for e in s]

def type_name(e, i):
    return {
        "U": "usize",
        "N": "NewAxis",
    }[e]

def component_value_usize(e, i, l, var="self"):
    return {
        "U": f"{var}.{l - i - 1}",
    }[e]

def shape_type(s):
    return "(" + "".join([type_name(e, i) + ", " for (i, e) in reversed(list(enumerate(s)))]) + ")"

def index_len(s):
    return len([None for e in s if e != "N"])

def impl_shape(s):
    index_val = "[" + "".join([component_value_usize(e, i, len(s)) + ", " for (i, e) in reversed(list(enumerate(s))) if e != "N"]) + "]"

    print(f"""
impl AsIndex for {shape_type(s)}
{{
    type Index = [usize; {index_len(s)}];

    fn as_index(&self) -> Self::Index {{
        {index_val}
    }}
}}
impl Shape for {shape_type(s)} {{}}
    """)


def impl_shape_eq(s1, s2):
    if len(s1) != len(s2):
        return

    def dims_equal(e1, e2, i):
        if e1 == "N" and e2 == "N":
            return True
        elif e1 == "U" and e2 == "U":
            return f"self.{len(s1) - i - 1} == other.{len(s2) - i - 1}"
        else:
            return False

    shape_eq = [dims_equal(e1, e2, i) for (i, (e1, e2)) in reversed(list(enumerate(zip(s1, s2))))]
    if any(e is False for e in shape_eq):
        return

    shape_eq = " && ".join([e for e in shape_eq if e is not True])
    if not shape_eq:
        shape_eq = "true"

    other = "other" if "other" in shape_eq else "_other"

    print(f"""
impl ShapeEq<{shape_type(s2)}> for {shape_type(s1)}
{{
    fn shape_eq(&self, {other}: &{shape_type(s2)}) -> bool {{
        {shape_eq}
    }}
}}
""")

def broadcast(s1, s2):
    def broadcast_dim(e1, e2):
        if e1 == "U" or e2 == "U":
            return "U"
        return "N"

    p1, p2 = pad(s1, s2)
    return "".join([broadcast_dim(e1, e2) for(e1, e2) in zip(p1, p2)])

def impl_into_index(s1, s2):
    if s2 != broadcast(s1, s2):
        return

    p1, p2 = pad(s1, s2)

    index_map = []
    for (i, (e1, e2)) in enumerate(zip(p1, p2)):
        if e1 != "N" and e2 != "N":
            index_map.append(f"index[{len(s1) - i - 1}], ")
        elif e1 == "N" and e2 == "N":
            pass
        elif e1 == "N" and e2 != "N":
            index_map.append("0, ")
        else:
            assert False, "Bad broadcast?"

    index_map_code = "[" + "".join(index_map) + "]"

    index = "index" if "index" in index_map_code else "_index"

    print(f"""
impl IntoIndex<{shape_type(s2)}> for {shape_type(s1)}
{{
    fn into_index({index}: [usize; {index_len(s1)}]) -> [usize; {index_len(s2)}] {{
        {index_map_code}
    }}
}}
""")


def impl_broadcast(s1, s2):
    s3 = broadcast(s1,s2)

    p1, p2, p3 = pad(s1, s2, s3)

    def dims_broadcast(e1, e2, i):
        if e1 == "N" and e2 == "N":
            return "NewAxis, "
        elif e1 == "U" and e2 == "N":
            return f"self.{len(s1) - i - 1}, "
        elif e1 == "N" and e2 == "U":
            return f"other.{len(s2) - i - 1}, "
        elif e1 == "U" and e2 == "U":
            return f"(self.{len(s1) - i - 1} == other.{len(s2) - i - 1}).then_some(self.{len(s1) - i - 1})?, "
        else:
            assert False, "Non-exhaustive match"

    shape_broadcast = [dims_broadcast(e1, e2, i) for (i, (e1, e2)) in reversed(list(enumerate(zip(p1, p2))))]
    if any(e is None for e in shape_broadcast):
        return

    shape_broadcast = "(" + "".join(shape_broadcast) + ")"

    other = "other" if "other" in shape_broadcast else "_other"

    print(f"""
impl BroadcastShape<{shape_type(s2)}> for {shape_type(s1)}
{{
    type Output = {shape_type(s3)};
    fn broadcast_shape(self, {other}: {shape_type(s2)}) -> Option<Self::Output> {{
        Some({shape_broadcast})
    }}
}}
""")


all_s = all_shapes(max_dim=2)

print("""
use crate::{AsIndex, Const, NewAxis, Shape, ShapeEq, IntoIndex, BroadcastShape};
""")


print("n=", len(all_s), file=sys.stderr)
print("n^2=", len(all_s)**2, file=sys.stderr)

for s in all_s:
    impl_shape(s)

print("//////////////////////////////")

for s1 in all_s:
    for s2 in all_s:
        impl_shape_eq(s1, s2)
        impl_into_index(s1, s2)
        impl_broadcast(s1, s2)
