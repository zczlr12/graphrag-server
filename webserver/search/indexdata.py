"""索引."""
import os
from typing import Optional, cast

import pandas as pd

from webserver import const

pd.set_option("display.max_columns", None)


async def get_index_data(input_dir: str, datatype: str, idx: int) -> pd.Series:
    document_df = pd.read_parquet(f"{input_dir}/{const.DOCUMENT_TABLE}.parquet")
    entity_df = pd.read_parquet(f"{input_dir}/{const.ENTITY_TABLE}.parquet")
    text_unit_df = pd.read_parquet(f"{input_dir}/{const.TEXT_UNIT_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{input_dir}/{const.ENTITY_EMBEDDING_TABLE}.parquet")
    relationship_df = pd.read_parquet(f"{input_dir}/{const.RELATIONSHIP_TABLE}.parquet")
    community_df = pd.read_parquet(f"{input_dir}/{const.COMMUNITY_TABLE}.parquet")
    report_df = pd.read_parquet(f"{input_dir}/{const.COMMUNITY_REPORT_TABLE}.parquet")
    if datatype == "documents":
        return await get_doc(document_df, idx)
    elif datatype == "entities":
        return await get_entity(document_df, entity_df, text_unit_df, entity_embedding_df, report_df, idx)
    elif datatype == "claims":
        return await get_claim(input_dir, idx)
    elif datatype == "sources":
        return await get_source(document_df, text_unit_df, entity_embedding_df, relationship_df, idx)
    elif datatype == "reports":
        return await get_report(document_df, text_unit_df, relationship_df, community_df, report_df, idx)
    elif datatype == "relationships":
        return await get_relationship(document_df, text_unit_df, entity_embedding_df, relationship_df, idx)
    else:
        raise ValueError(f"Unknown datatype: {datatype}")


async def get_doc(document_df: pd.DataFrame, idx: int) -> pd.Series:
    return document_df.iloc[idx]


# async def get_entity(input_dir: str, row_id: Optional[int] = None) -> Entity:
#     entity_df = pd.read_parquet(f"{input_dir}/{consts.ENTITY_TABLE}.parquet")
#     entity_embedding_df = pd.read_parquet(f"{input_dir}/{consts.ENTITY_EMBEDDING_TABLE}.parquet")

#     entities = read_indexer_entities(entity_df, entity_embedding_df, consts.COMMUNITY_LEVEL)
#     # TODO optimize performance using like database or dict in memory
#     for entity in entities:
#         if int(entity.short_id) == row_id:
#             return entity
#     raise ValueError(f"Not Found entity id {row_id}")


async def get_entity(
    document_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    text_unit_df: pd.DataFrame,
    entity_embedding_df: pd.DataFrame,
    report_df: pd.DataFrame,
    idx: int
) -> pd.Series:
    """提取实体信息."""
    entity_df = cast(
        pd.DataFrame,
        entity_df[["level", "title", "degree", "community", "x", "y"]]
    )
    entity_embedding_df.rename(
        columns={"name": "title"},
        inplace=True
    )
    entity_df = entity_df.merge(entity_embedding_df, on="title", how="inner")
    entities = entity_df[entity_df["human_readable_id"] == idx]
    entity = entities.iloc[0]
    entity["communities"] = {}
    for row in entities.itertuples():
        entity["communities"][row.level] = (
            row.community,
            report_df[
                report_df["community"] == row.community
            ].iloc[0]["title"]
        ) if row.community else None
    entity["sources"] = {}
    for source_id in entity["text_unit_ids"]:
        text_unit = text_unit_df[text_unit_df["id"] == source_id]
        title = document_df[
            document_df["id"] == text_unit["document_ids"].iloc[0][0]
        ].iloc[0]["title"]
        if not entity["sources"].get(title):
            entity["sources"][title] = []
        entity["sources"][title].append(text_unit.index[0])
    for source_list in entity["sources"].values():
        source_list.sort()
    return entity


# async def get_claim(input_dir: str, row_id: Optional[int] = None) -> Covariate:
#     covariate_file = f"{input_dir}/{consts.COVARIATE_TABLE}.parquet"
#     if os.path.exists(covariate_file):
#         covariate_df = pd.read_parquet(covariate_file)
#         claims = read_indexer_covariates(covariate_df)
#     else:
#         raise ValueError(f"No claims {input_dir} of id {row_id} found")
#     # TODO optimize performance using like database or dict in memory
#     for claim in claims:
#         if int(claim.short_id) == row_id:
#             return claim
#     raise ValueError(f"Not Found claim id {row_id}")


async def get_claim(input_dir: str, row_id: Optional[int] = None) -> pd.Series:
    covariate_file = f"{input_dir}/{const.COVARIATE_TABLE}.parquet"
    if os.path.exists(covariate_file):
        covariate_df = pd.read_parquet(covariate_file)
        return covariate_df[covariate_df["human_readable_id"] == str(row_id)].iloc[0]
    raise ValueError(f"No claims {input_dir} of id {row_id} found")


# async def get_source(input_dir: str, row_id: Optional[int] = None) -> TextUnit:
#     text_unit_df = pd.read_parquet(f"{input_dir}/{consts.TEXT_UNIT_TABLE}.parquet")
#     text_units = read_indexer_text_units(text_unit_df)
#     # TODO optimize performance using like database or dict in memory
#     for text_unit in text_units:
#         if int(text_unit.short_id) == row_id:
#             return text_unit
#     raise ValueError(f"Not Found source id {row_id}")


async def get_source(
    document_df: pd.DataFrame,
    text_unit_df: pd.DataFrame,
    entity_embedding_df: pd.DataFrame,
    relationship_df: pd.DataFrame,
    row_id: int
) -> pd.Series:
    text_unit = text_unit_df.rename(
        columns={
            "document_ids": "documents",
            "entity_ids": "entities",
            "relationship_ids": "relationships"
        }
    ).iloc[row_id]
    for i, id in enumerate(text_unit["documents"]):
        index = document_df[document_df["id"] == id].index[0]
        document = document_df[document_df["id"] == id].iloc[0]
        text_unit["documents"][i] = (index, document["title"])
    for i, id in enumerate(text_unit["entities"]):
        entity = entity_embedding_df[entity_embedding_df["id"] == id].iloc[0]
        text_unit["entities"][i] = (entity["human_readable_id"], entity["name"])
    text_unit["entities"].sort()
    for i, id in enumerate(text_unit["relationships"]):
        relationship = relationship_df[relationship_df["id"] == id].iloc[0]
        text_unit["relationships"][i] = (int(relationship["human_readable_id"]), f"{relationship['source']} ↔ {relationship['target']}")
    text_unit["relationships"].sort()
    text_unit["title"] = text_unit["text"].replace("\n", "")
    if len(text_unit["title"]) > 50:
        text_unit["title"] = text_unit["title"][:50] + "..."
    return text_unit


# async def get_report(input_dir: str, row_id: Optional[int] = None) -> CommunityReport:
#     entity_df = pd.read_parquet(f"{input_dir}/{consts.ENTITY_TABLE}.parquet")
#     report_df = pd.read_parquet(f"{input_dir}/{consts.COMMUNITY_REPORT_TABLE}.parquet")
#     reports = read_indexer_reports(report_df, entity_df, consts.COMMUNITY_LEVEL)
#     # TODO optimize performance using like database or dict in memory
#     for report in reports:
#         if int(report.short_id) == row_id:
#             return report
#     raise ValueError(f"Not Found report id {row_id}")


async def get_report(
    document_df: pd.DataFrame,
    text_unit_df: pd.DataFrame,
    relationship_df: pd.DataFrame,
    community_df: pd.DataFrame,
    report_df: pd.DataFrame,
    row_id: int
) -> pd.Series:
    report_df = report_df.merge(
        community_df.rename(columns={"id": "community"})[["community", "relationship_ids", "text_unit_ids"]],
        on="community",
        how="inner"
    ).rename(columns={"relationship_ids": "relationships"})
    report = report_df[report_df["community"].astype(int) == row_id].iloc[0]
    for i, relationship_id in enumerate(report["relationships"]):
        relationship = relationship_df[relationship_df["id"] == relationship_id].iloc[0]
        report["relationships"][i] = (int(relationship["human_readable_id"]), f"{relationship['source']} ↔ {relationship['target']}")
    report["relationships"].sort()
    report["sources"] = {}
    for source_id in report["text_unit_ids"][0].split(","):
        text_unit = text_unit_df[text_unit_df["id"] == source_id]
        title = document_df[
            document_df["id"] == text_unit["document_ids"].iloc[0][0]
        ].iloc[0]["title"]
        if not report["sources"].get(title):
            report["sources"][title] = []
        report["sources"][title].append(text_unit.index[0])
    for source_list in report["sources"].values():
        source_list.sort()
    return report


# async def get_relationship(input_dir: str, row_id: Optional[int] = None) -> Relationship:
#     relationship_df = pd.read_parquet(f"{input_dir}/{consts.RELATIONSHIP_TABLE}.parquet")
#     relationships = read_indexer_relationships(relationship_df)
#     # TODO optimize performance using like database or dict in memory
#     for relationship in relationships:
#         if int(relationship.short_id) == row_id:
#             return relationship
#     raise ValueError(f"Not Found relationship id {row_id}")


async def get_relationship(
    document_df: pd.DataFrame,
    text_unit_df: pd.DataFrame,
    entity_embedding_df: pd.DataFrame,
    relationship_df: pd.DataFrame,
    row_id: int
) -> pd.Series:
    relationship = relationship_df[relationship_df["human_readable_id"] == str(row_id)].iloc[0]
    relationship["source"] = (
        entity_embedding_df[
            entity_embedding_df["name"] == relationship["source"]
        ].iloc[0]["human_readable_id"],
        relationship["source"]
    )
    relationship["target"] = (
        entity_embedding_df[
            entity_embedding_df["name"] == relationship["target"]
        ].iloc[0]["human_readable_id"],
        relationship["target"]
    )
    relationship["sources"] = {}
    for source_id in relationship["text_unit_ids"]:
        text_unit = text_unit_df[text_unit_df["id"] == source_id]
        title = document_df[
            document_df["id"] == text_unit["document_ids"].iloc[0][0]
        ].iloc[0]["title"]
        if not relationship["sources"].get(title):
            relationship["sources"][title] = []
        relationship["sources"][title].append(text_unit.index[0])
        relationship["sources"][title].sort()
    relationship["title"] = f"{relationship['source'][1]} ↔ {relationship['target'][1]}"
    return relationship
