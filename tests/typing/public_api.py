"""Static consumer contract; pyright checks this file without running it."""

from typing import assert_type

import pandas as pd

import fieldwork as fw
from fieldwork.navigation import Path
from fieldwork.typing import DiscoveryOptions, SectionOptions


def consume(df: pd.DataFrame) -> None:
    events: list[fw.ProgressEvent] = []
    token = fw.CancellationToken()
    scope = fw.Scope.from_positions(df, [0], progress=events.append)
    assert_type(scope, fw.Scope)
    assert_type(scope.refine(df, []), fw.Scope)
    assert_type(token.cancelled, bool)
    options: DiscoveryOptions = {"objective": "availability", "max_candidates": 20}
    sections: SectionOptions = {"dependencies": {"include_grain": False}}
    overview = fw.explore(
        df,
        discovery=options,
        section_options=sections,
        progress=events.append,
        cancel=token,
        timeout=5,
        scope=scope,
    )
    assert_type(overview, fw.InvestigationResult)
    assert_type(overview.to_frame(), pd.DataFrame)
    assert_type(overview.inspect(df, "f0", all_matches=True), pd.DataFrame)
    assert_type(overview.select(df, "f0"), fw.Scope)
    assert_type(overview.relationships("site"), pd.DataFrame)
    assert_type(fw.explore(df, ["site"], candidate_keys=["site"]), fw.ExplorerResult)
    assert_type(fw.census(df, ["site"], top_n=2, progress=True), fw.ExplorerResult)
    assert_type(fw.levels(df, ["site"], timeout=2), fw.ExplorerResult)
    assert_type(fw.grain(df, [fw.KeySpec("site", ("site",))]), fw.ExplorerResult)
    assert_type(fw.pairs(df, ["site", "visit"], max_pairs=1), fw.ExplorerResult)
    assert_type(fw.joint_counts(df, ["site", "visit"]), fw.ExplorerResult)
    assert_type(fw.infer_schema(df), fw.ExplorerResult)
    missingness = fw.missingness(df, unit="entities", entity="site")
    assert_type(missingness, fw.InvestigationResult)
    assert_type(fw.compare(missingness, missingness), fw.InvestigationResult)
    assert_type(fw.discover_dependencies(df), fw.InvestigationResult)
    assert_type(fw.value_patterns(df), fw.InvestigationResult)
    paths = fw.suggest_paths(df, objective="structure")
    assert_type(paths, fw.PathResult)
    assert_type(paths.best, Path | None)
    best = paths.best
    if best is not None:
        assert_type(best.dimensions, tuple[str, ...])
        assert_type(best.census(df, top_n=2, progress=True), fw.ExplorerResult)
    assert_type(paths.path(), Path)
    assert_type(fw.render_plaintext(overview), str)
    assert_type(fw.render_svg(overview, view="findings"), str)
    assert_type(fw.render_html(overview, detail="topology"), str)
    assert_type(fw.ExplorerResult.from_dict(fw.levels(df).to_dict()), fw.ExplorerResult)
    assert_type(fw.InvestigationResult.from_dict(overview.to_dict()), fw.InvestigationResult)
    assert_type(fw.Recipe("missingness").run(df, progress=True), fw.ExplorerResult)
    assert_type(fw.Recipe.load("recipe.json"), fw.Recipe)

    # Expected failures: suppress only the specific diagnostic. With unused
    # ignore reporting, accepting any of these becomes a failing contract.
    fw.missingness(df, unit="people")  # pyright: ignore[reportArgumentType]
    fw.explore(df, candidate_keys=["site"])  # pyright: ignore[reportCallIssue]
    fw.explore(df, ["site"], sections=["paths"])  # pyright: ignore[reportArgumentType]
    fw.census(df, ["site"], include_pairs=True)  # pyright: ignore[reportCallIssue]
    fw.grain(df, ["site"], _cache=None)  # pyright: ignore[reportCallIssue]
    paths.path().census(df, missing={})  # pyright: ignore[reportCallIssue]
    fw.render_svg(overview, view="bogus")  # pyright: ignore[reportArgumentType]
