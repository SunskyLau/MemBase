"""按版本、读取时点和来源片段去重输出；不裁剪证明内部的必要前提。"""
from __future__ import annotations
from .persistence import canonical_json


def render_evidence(bundles, view, evaluation, policy=None):
    nodes, links, sources = {}, {}, {}
    for bundle in bundles:
        for node in bundle.nodes:
            nodes[(node["version_id"], canonical_json(node["at_time"]))] = node
        for link in bundle.links:
            links[(link["dependency_id"], canonical_json(link["at_time"]))] = link
        for ref in bundle.refs:
            if ref.type == "SOURCE":
                for span in [ref.span, *ref.context_refs]:
                    sources[(span.source_id, span.start, span.end)] = span
    aliases = {key: f"M{i + 1}" for i, key in enumerate(nodes)}
    source_aliases = {key: f"S{i + 1}" for i, key in enumerate(sources)}
    lines = ["CURRENT statements are usable at the query snapshot/time. HISTORICAL statements are not current values.",
             "Source order is input order, not an invented event date. Planned/uncertain statements retain that meaning."]
    if (policy or {}).get("update_priority") == "newer_source":
        lines.append("For conflicting values of the same subject/property, use the larger source order, even when it differs from real-world knowledge.")
    for key, node in nodes.items():
        version = view.versions[node["version_id"]]
        current = (evaluation.evaluate(version.id).usable
                   and canonical_json(node["at_time"]) == canonical_json(evaluation.now))
        label = "CURRENT" if current else "HISTORICAL"
        time = version.valid_time
        detail = f"; modality={version.modality}" if version.modality != "asserted" else ""
        if time.start or time.end or time.text:
            detail += f"; validity={canonical_json(time)}"
        if not current:
            detail += f"; supported_at={canonical_json(node['at_time'])}"
        lines.append(f"{aliases[key]} [{label}{detail}]: {version.content}")
    for key, link in links.items():
        dependency = view.dependencies[link["dependency_id"]]
        target = aliases.get((dependency.target_version_id, key[1]))
        if target is None:
            continue
        premises = []
        for ref in dependency.premise_refs:
            if ref.type == "SOURCE":
                premises.extend(source_aliases[(span.source_id, span.start, span.end)]
                                for span in [ref.span, *ref.context_refs])
            else:
                point = ref.at_time.model_dump(mode="json") if ref.type == "HISTORICAL" else link["at_time"]
                premises.append(aliases[(ref.id, canonical_json(point))])
        lines.append(f"Support {target}: " + " AND ".join(dict.fromkeys(premises)))
    for key, span in sources.items():
        source = view.sources[span.source_id]
        quote = source.content[span.start:span.end]
        lines.append(f"{source_aliases[key]} [source order={source.source_order}; speaker={source.speaker}; "
                     f"date={source.mention_time or 'unknown'}]: {quote}")
    return "\n".join(lines)
