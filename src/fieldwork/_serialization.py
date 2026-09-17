"""Opt-in, versioned JSON envelope for shared result containers."""

from __future__ import annotations

FORMAT = "fieldwork.compact"


def compact_result(data):
    counts, active = {}, set()

    def count(value):
        if not isinstance(value, (dict, list, tuple)):
            return
        key = id(value)
        if key in active:
            raise ValueError("Cyclic results cannot be exported")
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > 1:
            return
        active.add(key)
        for child in value.values() if isinstance(value, dict) else value:
            count(child)
        active.remove(key)

    count(data)
    objects, indices = [], {}

    def content(value):
        if isinstance(value, dict):
            if "$ref" in value or "$dict" in value:
                return {"$dict": [[k, encode(v)] for k, v in value.items()]}
            return {k: encode(v) for k, v in value.items()}
        return [encode(v) for v in value]

    def encode(value):
        if not isinstance(value, (dict, list, tuple)):
            return value
        # Tiny containers cost less to repeat than to reference.
        if counts[id(value)] > 1 and len(value) > 3:
            key = id(value)
            if key not in indices:
                indices[key] = len(objects)
                objects.append(None)
                objects[indices[key]] = content(value)
            return {"$ref": indices[key]}
        return content(value)

    root = encode(data)
    return {"format": FORMAT, "version": "1.0", "root": root, "objects": objects}


def expand_result(data):
    if data.get("format") != FORMAT:
        return data
    if (
        data.get("version") != "1.0"
        or not isinstance(data.get("objects"), list)
        or "root" not in data
    ):
        raise ValueError("Unsupported compact result envelope")
    objects, resolved, active = data["objects"], {}, set()

    def decode(value):
        if isinstance(value, list):
            return [decode(v) for v in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"$ref"}:
            index = value["$ref"]
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(objects)
            ):
                raise ValueError("Invalid compact result reference")
            if index in active:
                raise ValueError("Cyclic compact result reference")
            if index not in resolved:
                active.add(index)
                resolved[index] = decode(objects[index])
                active.remove(index)
            return resolved[index]
        if set(value) == {"$dict"}:
            pairs = value["$dict"]
            if not isinstance(pairs, list) or any(
                not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str)
                for pair in pairs
            ):
                raise ValueError("Invalid escaped compact dictionary")
            return {k: decode(v) for k, v in pairs}
        return {k: decode(v) for k, v in value.items()}

    root = decode(data["root"])
    if not isinstance(root, dict):
        raise TypeError("Compact result root must be an object")
    return root
