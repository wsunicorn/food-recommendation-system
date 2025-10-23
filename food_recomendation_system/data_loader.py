import os
import re
import html
import pandas as pd


def load_data(path=None):
    """Load dataset from CSV. By default reads from the workspace data folder.

    Returns:
        pd.DataFrame
    """
    if path is None:
        base = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(base, "data", "Dataset_for_print.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    # read CSV with low_memory to avoid dtype guessing issues
    df = pd.read_csv(path, low_memory=True)

    # Normalize common columns to 'title' and 'description'
    if 'title' not in df.columns:
        if 'name_food' in df.columns:
            df = df.rename(columns={'name_food': 'title'})
        elif 'name' in df.columns:
            df = df.rename(columns={'name': 'title'})

    if 'description' not in df.columns:
        # prefer user reviews as description if available
        if 'review_user' in df.columns:
            df['description'] = df['review_user'].astype(str)
        elif 'ingredients' in df.columns:
            df['description'] = df['ingredients'].astype(str)
        else:
            # fallback: concatenate all object columns except title
            text_cols = [c for c in df.columns if df[c].dtype == object and c != 'title']
            if text_cols:
                df['description'] = df[text_cols].astype(str).agg(' '.join, axis=1)
            else:
                df['description'] = df.get('title', '').astype(str)

        # clean description: remove HTML tags (like <br/>), unescape HTML entities, normalize whitespace
        def clean_text(s: str) -> str:
            if s is None:
                return ''
            if not isinstance(s, str):
                s = str(s)
            # remove HTML tags
            s = re.sub(r'<[^>]+>', ' ', s)
            # unescape HTML entities
            s = html.unescape(s)
            # collapse whitespace
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        df['description'] = df['description'].fillna('').apply(clean_text)

        # Add a simple category field extracted from ingredients or title
    def infer_category(row):
        txt = ''
        if pd.notna(row.get('ingredients')):
            txt = str(row.get('ingredients')).lower()
        else:
            txt = str(row.get('title', '')).lower()
        # simple keyword-based categories
        if any(k in txt for k in ('chicken', 'chick')):
            return 'Chicken'
        if any(k in txt for k in ('beef', 'steak')):
            return 'Beef'
        if any(k in txt for k in ('fish', 'tilapia', 'salmon', 'tuna', 'cod', 'sea')):
            return 'Fish'
        if any(k in txt for k in ('pork', 'bacon', 'ham')):
            return 'Pork'
        if any(k in txt for k in ('vegetable', 'veggie', 'tofu', 'salad', 'cabbage')):
            return 'Vegetarian'
        if any(k in txt for k in ('rice', 'noodle', 'spaghetti', 'pasta')):
            return 'Rice/Pasta'
        if any(k in txt for k in ('sauce', 'stir-fry', 'stir fry')):
            return 'Sauce/Condiment'
        return 'Other'

    if 'category' not in df.columns:
        df['category'] = df.apply(infer_category, axis=1)

    return df