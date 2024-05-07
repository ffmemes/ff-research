import numpy as np
import polars as pl

from models import BaseRecommender


def estimate_one(recs, target_memes, target_reactions):
    """Matches recs with future seen memes from target list
    Calculates likes and dislikes"""
    likes = 0
    dislikes = 0
    for meme_id in recs:
        if meme_id not in target_memes:
            continue
        idx = target_memes.index(meme_id)
        reaction = target_reactions[idx]
        if reaction == 1:
            likes += 1
            continue
        if reaction == 2:
            dislikes += 1
            continue

    if (likes + dislikes) == 0:
        return None, None, None
    
    lr = likes / (likes + dislikes)

    return likes, dislikes, lr


def print_results(results_df: pl.DataFrame):
    likes = results_df['likes'].sum()
    lr = results_df['likes'].sum() / (results_df['likes'].sum() + results_df['dislikes'].sum())
    lr_micro = results_df['lr'].mean()
    btstrp_lr_micro = []
    btstrp_lr_macro = []
    n = len(results_df)
    for _ in range(1000):
        btsrp_results_df = results_df.sample(n, with_replacement=True)
        btstrp_lr_micro.append(btsrp_results_df['lr'].mean())
        btstrp_lr_macro.append(btsrp_results_df['likes'].sum() / (btsrp_results_df['likes'].sum() + btsrp_results_df['dislikes'].sum()))
    btstrp_micro_std = np.std(btstrp_lr_micro)
    btstrp_macro_std = np.std(btstrp_lr_macro)

    print(f'Likes - {likes}, Like Rate = {lr:.3f} +- {btstrp_macro_std * 1.96:.3f}, Like Rate Micro = {lr_micro:.3f} +- {btstrp_micro_std * 1.96:.3f}')


def estimate(model: BaseRecommender, df: pl.DataFrame, lang_code: str = None, top_size=100):
    """Estimates the model based on top_size recs"""
    rows = []
    for row in df.iter_rows(named=True):
        recs = model.recommend(row['user_id'], row['date_dtm'], row['hist_memes'], row['hist_reactions'], lang_code=lang_code)[:top_size]

        likes, dislikes, lr = estimate_one(recs, row['target_memes'], row['target_reactions'])

        rows.append({
            'user_id': row['user_id'],
            'hist_size': row['hist_size'],
            'date_dtm': row['date_dtm'],
            'likes': likes,
            'dislikes': dislikes,
            'lr': lr,
        })

    results_df = pl.DataFrame(rows)
    print_results(results_df)



def estimate_prod(recommended_by: str, df: pl.DataFrame):
    """Estimates production models"""
    rows = []
    for row in df.iter_rows(named=True):
        recs = [meme_id for meme_id, _recommended_by in zip(row['target_memes'], row['target_recommended_by']) if _recommended_by == recommended_by]

        likes, dislikes, lr = estimate_one(recs, row['target_memes'], row['target_reactions'])

        rows.append({
            'user_id': row['user_id'],
            'hist_size': row['hist_size'],
            'date_dtm': row['date_dtm'],
            'likes': likes,
            'dislikes': dislikes,
            'lr': lr,
        })

    results_df = pl.DataFrame(rows)
    print_results(results_df)