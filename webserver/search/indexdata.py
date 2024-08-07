import os
from typing import Optional, cast

import pandas as pd

from graphrag.model import Relationship, Covariate, Entity, CommunityReport, TextUnit
from graphrag.query.indexer_adapters import read_indexer_relationships, read_indexer_covariates, read_indexer_entities, \
    read_indexer_reports, read_indexer_text_units
from ..utils import consts


async def get_index_data(input_dir: str, datatype: str, id: Optional[int] = None):
    entity_df = pd.read_parquet(f"{input_dir}/{consts.ENTITY_TABLE}.parquet")
    text_unit_df = pd.read_parquet(f"{input_dir}/{consts.TEXT_UNIT_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{input_dir}/{consts.ENTITY_EMBEDDING_TABLE}.parquet")
    report_df = pd.read_parquet(f"{input_dir}/{consts.COMMUNITY_REPORT_TABLE}.parquet")
    if datatype == "entities":
        return await get_entity(entity_df, text_unit_df, entity_embedding_df, report_df, id)
    elif datatype == "claims":
        return await get_claim(input_dir, id)
    elif datatype == "sources":
        return await get_source(input_dir, id)
    elif datatype == "reports":
        return await get_report(input_dir, id)
    elif datatype == "relationships":
        return await get_relationship(input_dir, id)
    else:
        raise ValueError(f"Unknown datatype: {datatype}")


# async def get_entity(input_dir: str, row_id: Optional[int] = None) -> Entity:
#     entity_df = pd.read_parquet(f"{input_dir}/{consts.ENTITY_TABLE}.parquet")
#     entity_embedding_df = pd.read_parquet(f"{input_dir}/{consts.ENTITY_EMBEDDING_TABLE}.parquet")

#     entities = read_indexer_entities(entity_df, entity_embedding_df, consts.COMMUNITY_LEVEL)
#     # TODO optimize performance using like database or dict in memory
#     for entity in entities:
#         if int(entity.short_id) == row_id:
#             return entity
#     raise ValueError(f"Not Found entity id {row_id}")


async def get_entity(entity_df: pd.DataFrame, text_unit_df: pd.DataFrame, entity_embedding_df: pd.DataFrame, report_df: pd.DataFrame, row_id: Optional[int] = None) -> dict:
    entity_df = cast(pd.DataFrame, entity_df[["title", "degree", "community"]]).rename(
        columns={"title": "name", "degree": "rank"}
    )

    entity_df["community"] = entity_df["community"].fillna(-1)
    entity_df["community"] = entity_df["community"].astype(int)
    entity_df["rank"] = entity_df["rank"].astype(int)

    # for duplicate entities, keep the one with the highest community level
    entity_df = (
        entity_df.groupby(["name", "rank"]).agg({"community": "max"}).reset_index()
    )
    entity_df["community"] = entity_df["community"].apply(lambda x: [str(x)])
    entity_df = entity_df.merge(
        entity_embedding_df, on="name", how="inner"
    ).drop_duplicates(subset=["name"])
    entity = entity_df[entity_df["human_readable_id"] == row_id].to_dict("records")[0]
    for i, short_id in enumerate(entity["community"]):
        community = report_df[report_df["community"] == short_id]
        entity["community"][i] = (community["community"].to_numpy()[0], community["title"].to_numpy()[0])
    for i, id in enumerate(entity["text_unit_ids"]):
        text_unit = text_unit_df[text_unit_df["id"] == id]
        entity["text_unit_ids"][i] = (text_unit.index[0], id)
    # read entity dataframe to knowledge model objects
    return entity


async def get_claim(input_dir: str, row_id: Optional[int] = None) -> Covariate:
    covariate_file = f"{input_dir}/{consts.COVARIATE_TABLE}.parquet"
    if os.path.exists(covariate_file):
        covariate_df = pd.read_parquet(covariate_file)
        claims = read_indexer_covariates(covariate_df)
    else:
        raise ValueError(f"No claims {input_dir} of id {row_id} found")
    # TODO optimize performance using like database or dict in memory
    for claim in claims:
        if int(claim.short_id) == row_id:
            return claim
    raise ValueError(f"Not Found claim id {row_id}")


async def get_source(input_dir: str, row_id: Optional[int] = None) -> TextUnit:
    text_unit_df = pd.read_parquet(f"{input_dir}/{consts.TEXT_UNIT_TABLE}.parquet")
    text_units = read_indexer_text_units(text_unit_df)
    # TODO optimize performance using like database or dict in memory
    for text_unit in text_units:
        if int(text_unit.short_id) == row_id:
            return text_unit
    raise ValueError(f"Not Found source id {row_id}")


async def get_report(input_dir: str, row_id: Optional[int] = None) -> CommunityReport:
    entity_df = pd.read_parquet(f"{input_dir}/{consts.ENTITY_TABLE}.parquet")
    report_df = pd.read_parquet(f"{input_dir}/{consts.COMMUNITY_REPORT_TABLE}.parquet")
    reports = read_indexer_reports(report_df, entity_df, consts.COMMUNITY_LEVEL)
    # TODO optimize performance using like database or dict in memory
    for report in reports:
        if int(report.short_id) == row_id:
            return report
    raise ValueError(f"Not Found report id {row_id}")


async def get_relationship(input_dir: str, row_id: Optional[int] = None) -> Relationship:
    relationship_df = pd.read_parquet(f"{input_dir}/{consts.RELATIONSHIP_TABLE}.parquet")
    relationships = read_indexer_relationships(relationship_df)
    # TODO optimize performance using like database or dict in memory
    for relationship in relationships:
        if int(relationship.short_id) == row_id:
            return relationship
    raise ValueError(f"Not Found relationship id {row_id}")
