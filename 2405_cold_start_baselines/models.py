from datetime import datetime
from typing import List

import polars as pl


class ColdStartRecommender():
    score = pl.when(pl.col('age_days') < 14).then(1.0).otherwise(0.8) * pl.col('n_likes') / (pl.col('n_likes') + pl.col('n_dislikes'))
    max_k = 1000
    shuffle = True

    """Uses no history and personalization"""
    def __init__(self, meme_features_daily_df: pl.DataFrame, min_sent_thr: int = 20):
        self._cache = dict()
        self.meme_features_daily_df = (
            meme_features_daily_df
            .filter(pl.col('n_memes_sent') > min_sent_thr)
            .with_columns(self.score.alias('score'))
            .sort('score', descending=True)
        )

    def _recommend(self, k: int, date_dtm: datetime, lang_codes: List[str] = None) -> pl.DataFrame:
        cached_df = self._cache[date_dtm]
        
        if lang_codes is not None:
            cached_df = cached_df.filter(pl.col('language_code').is_in(lang_codes))

        cached_df = cached_df.head(k)

        if self.shuffle:
            cached_df = cached_df.sample(len(cached_df), shuffle=True, seed=int(date_dtm.timestamp()))

        return cached_df
        
    def _get_top_k_from_source(self, source: str, k: int, date_dtm: datetime) -> pl.DataFrame:
        return (
            self.meme_features_daily_df
            .filter(pl.col('date_dtm') == date_dtm)
            .filter(pl.col('url') == source)
            .head(k)
        )
    
    def recommend(self, k: int, date_dtm: datetime, lang_codes: List[str] = None) -> List[int]:
        recs_df = self._recommend(k, date_dtm, lang_codes=lang_codes)
        return (
            recs_df
            .get_column('meme_id')
            .to_list()
        )



class BestMemeFromEachSource(ColdStartRecommender):
    """
    Similar to production. Simplifications:
    Memes without stats were omitted  
    Top impression feature is omitted (gives 1.0 vs 0.8 for top 1 meme from a source by its telegram impressions)
    Impressions without reactions are omitted
    """

    def __init__(self, meme_features_daily_df, min_sent_thr=20):
        super().__init__(meme_features_daily_df, min_sent_thr)

        for date_dtm in meme_features_daily_df.select('date_dtm').unique().get_column('date_dtm').to_list():
            self._cache[date_dtm] = (
                meme_features_daily_df
                .filter(pl.col('date_dtm') == date_dtm)
                .filter(pl.col('n_likes') + pl.col('n_dislikes') > 0)
                .with_columns(self.score.alias('score'))
                .sort('score', descending=True)
                .group_by('meme_source_id')
                .agg(pl.all().first())
            )


class MostLiked(ColdStartRecommender):
    """
    Similar to production. Simplifications:
    Memes without stats were omitted  
    Top impression feature is omitted (gives 1.0 vs 0.8 for top 1 meme from a source by its telegram impressions)
    Impressions without reactions are omitted
    """

    def __init__(self, meme_features_daily_df, min_sent_thr=20):
        super().__init__(meme_features_daily_df, min_sent_thr)

        for date_dtm in meme_features_daily_df.select('date_dtm').unique().get_column('date_dtm').to_list():
            recs = (
                meme_features_daily_df
                .filter(pl.col('date_dtm') == date_dtm)
                .filter(pl.col('n_memes_sent') > 10)
                .with_columns(self.score.alias('score'))
            )

            self._cache[date_dtm] = (
                recs
                .sort('score', descending=True)
                .head(self.max_k)
            )


class CustomSources(ColdStartRecommender):
    """
    Returns top for specific sources
    """

    score = pl.when(pl.col('age_days') < 14).then(1.0).otherwise(0.8) * pl.col('n_likes') / (pl.col('n_likes') + pl.col('n_dislikes'))

    default_sources = [
        'tg://user?id=2034335896',
        'tg://user?id=670456638',
        'tg://user?id=1034761769',
        'tg://user?id=1485189091',
        'tg://user?id=49820636',
        'https://t.me/dailyepicmemes',
        'https://t.me/memekingdomtm',
        'https://t.me/dngut',
        'https://vk.com/mysterious_conditions',
        'https://vk.com/wtf.rasha',
        'https://vk.com/demotiva',
        'https://vk.com/papkapic',
        'https://t.me/ithumor',
        'https://t.me/profunctor_io',
        'https://t.me/programmerjokes',
        'https://vk.com/weirdanimals',
    ]

    def __init__(self, meme_features_daily_df, min_sent_thr=20, sources=None):
        super().__init__(meme_features_daily_df, min_sent_thr)
        self.sources = self.default_sources
        if sources is None:
            self.sources = sources

        for date_dtm in meme_features_daily_df.select('date_dtm').unique().get_column('date_dtm').to_list():
            recs = (
                meme_features_daily_df
                .filter(pl.col('date_dtm') == date_dtm)
                .filter(pl.col('n_memes_sent') > 10)
                .filter(pl.col('url').is_in(self.sources))
                .with_columns(self.score.alias('score'))
            )

            self._cache[date_dtm] = (
                recs
                .sort('score', descending=True)
                .head(self.max_k)
            )


class SelectedSources(ColdStartRecommender):
    """
    """

    top_users = [
        'tg://user?id=2034335896',
        'tg://user?id=670456638',
        'tg://user?id=1034761769',
        'tg://user?id=1485189091',
        'tg://user?id=49820636',
    ]

    general_en = [
        'https://t.me/dailyepicmemes',
        'https://t.me/memekingdomtm',
        'https://t.me/dngut',
    ]

    general_ru = [
        'https://vk.com/mysterious_conditions',
        'https://vk.com/wtf.rasha',
        'https://vk.com/demotiva',
        'https://vk.com/papkapic',
    ]

    it_ru = [
        'https://t.me/ithumor',
        'https://t.me/profunctor_io',
    ]

    it_en = [
        'https://t.me/programmerjokes',
    ]

    animals_ru = [
        'https://vk.com/weirdanimals',
    ]

    def __init__(self, meme_features_daily_df, min_sent_thr=20):
        super().__init__(meme_features_daily_df, min_sent_thr)

        for date_dtm in meme_features_daily_df.select('date_dtm').unique().get_column('date_dtm').to_list():
            candidates = []

            for source in self.top_users:
                candidates.append(self._get_top_k_from_source(source, 4, date_dtm)) 

            for source in self.general_ru:
                candidates.append(self._get_top_k_from_source(source, 10, date_dtm)) 

            for source in self.general_en:
                candidates.append(self._get_top_k_from_source(source, 10, date_dtm)) 

            for source in self.animals_ru:
                candidates.append(self._get_top_k_from_source(source, 5, date_dtm)) 

            # for source in self.it_ru:
            #     candidates.append(self._get_top_k_from_source(source, 4, date_dtm)) 

            # for source in self.it_en:
            #     candidates.append(self._get_top_k_from_source(source, 4, date_dtm)) 
            
            candidates_df = pl.concat(candidates)

            self._cache[date_dtm] = candidates_df.head(self.max_k)
    