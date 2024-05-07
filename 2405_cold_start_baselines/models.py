from datetime import datetime

import polars as pl


class BaseRecommender():
    def recommend(self, user_id, date_dtm, prev_ids, prev_reactions, lang_code=None):
        raise NotImplementedError

    def filter_seen(self, recs, hist_memes):
        return [meme_id for meme_id in recs if meme_id not in hist_memes]


class BestMemeFromEachSource(BaseRecommender):
    """
    Similar to production. Simplifications:
    Memes without stats were omitted  
    Top impression feature is omitted (gives 1.0 vs 0.8 for top 1 meme from a source by its telegram impressions)
    Impressions without reactions are omitted
    """

    score = pl.when(pl.col('age_days') < 14).then(1.0).otherwise(0.8) * pl.col('n_likes') / (pl.col('n_likes') + pl.col('n_dislikes'))

    def __init__(self, meme_features_daily_df):

        self._cache = dict()
        for date_dtm in meme_features_daily_df.select('date_dtm').unique().get_column('date_dtm').to_list():
            for lang_code in ['ru', 'en', None]:
                recs = (
                    meme_features_daily_df
                    .filter(pl.col('date_dtm') == date_dtm)
                    .filter(pl.col('n_likes') + pl.col('n_dislikes') > 0)
                    .with_columns(self.score.alias('score'))
                    .sort('score', descending=True)
                    .group_by('meme_source_id')
                    .agg(pl.all().first())
                )
                if lang_code is not None:
                    recs = recs.filter(pl.col('language_code') == lang_code)

                self._cache[(date_dtm, lang_code)] = (
                    recs
                    .get_column('meme_id')
                    .to_list()
                )


    def recommend(self, user_id, date_dtm, prev_ids, prev_reactions, lang_code=None):
        return self.filter_seen(self._cache[(date_dtm, lang_code)], prev_ids)


class MostLiked(BaseRecommender):
    """
    Similar to production. Simplifications:
    Memes without stats were omitted  
    Top impression feature is omitted (gives 1.0 vs 0.8 for top 1 meme from a source by its telegram impressions)
    Impressions without reactions are omitted
    """

    score = pl.when(pl.col('age_days') < 14).then(1.0).otherwise(0.8) * pl.col('n_likes') / (pl.col('n_likes') + pl.col('n_dislikes'))

    def __init__(self, meme_features_daily_df):

        self._cache = dict()

        for date_dtm in meme_features_daily_df.select('date_dtm').unique().get_column('date_dtm').to_list():
            for lang_code in ['ru', 'en', None]:
                recs = (
                    meme_features_daily_df
                    .filter(pl.col('date_dtm') == date_dtm)
                    .filter(pl.col('n_memes_sent') > 10)
                    .with_columns(self.score.alias('score'))
                )
                if lang_code is not None:
                    recs = recs.filter(pl.col('language_code') == lang_code)

                self._cache[(date_dtm, lang_code)] = (
                    recs
                    .sort('score', descending=True)
                    .head(100)
                    .select(pl.col('meme_id').shuffle(int(date_dtm.timestamp())))
                    .get_column('meme_id')
                    .to_list()
                )

    def recommend(self, user_id, date_dtm, prev_ids, prev_reactions, lang_code=None):
        return self.filter_seen(self._cache[(date_dtm, lang_code)], prev_ids)