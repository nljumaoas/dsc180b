import time

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer not started. Call start() before stop().")
        elapsed_time = time.time() - self.start_time
        self.start_time = None  # Reset timer
        return elapsed_time

def oma_text_isolator(book, page):
    """
    book: tojime no siora would expect om_a[0]
    page: page number (int)
    """
    page_field = book['pages'][page]
    page_text = []

    for text_field in page_field['text']:
        page_text.append(text_field['text_ja'])

    return page_text

['いや、それは．．．']

['こちらにいるレーネさんを', 
'迎えにあがったのですが．．．', 
'あっ兄貴',
'こいつさっき俺が報告した妙な閉じ眼だ！', 
'なんビル', 
'つけられたのか？', 
'つっつけようったって．．．', 
'こいつは眼が．．．', 
'匂いですよ', 
'レーネさんが作ったハンバーグの美味しそうな香り', 
'それを辿ってここまで来てみたら．．．']

['腐肉を煮詰めたような臭いが充満していて', 
'大変気分が悪い', 
'このままではせっかくの夕飯も冷めてしまう', 
'お互い銃を抜くことなく',
'どうか穏便に帰して頂けませんか', 
'スッ．．．', 
'ビル', 
'おう', 
'状況が全く見えてないマヌケには', 
'鉛弾で思い知らせてやらねばな', 
'閉じ眼如きが舐めた口ききやがって', 
'キャッ', 'シオラッ！！']

['くたばれ', '．．．', 'ファン', 'へ']

['．．．', '．．．', '１／１', '．．．', 'いや、', '撃ったのは', 'そっちが先ですよ']

['ヒルッ！？', '絶対逃がすな！！', '．．．', '！！', '！！', 'くそっ構えろ', '！！', 'ぶっ殺せ！！', 'あの、', '．．．っ！？当たらねぇ！！', '！！', '！！', 'ん', 'それでも、', 'どうなってやがる！！', 'そんな荒っぽく引き金を引いてちゃ', 'どこから飛んでくるか丸わかりだ', 'うおおっ！！', 'はーい！！', 'ひっ']


e_020 = ['自分がサインした契約書が',
 '多額の借用書だったなんて',
 '待てよ．．．じゃあ父さんは．．．',
 '視力を無くした画家になんの価値がある？',
 '可哀想だからあの世に送ってやったよ！',
 '今頃は天使のヌードでも描いてるだろうさ！！']

e_021 = ['あ？',
 'ててっ',
 '思ったより窓が高かった...',
 'おい．．',
 'なんだこいつ．．．',
 'シオラッ！？',
 'いやぁ',
 '夜分遅くにすいません．．．'
 ]

e_022 = ['こちらにいるレーネさんを', 
    '迎えにあがったのですが．．．', 
    'あっ兄貴',
    'こいつさっき俺が報告した妙な閉じ眼だ！', 
    'なんビル', 
    'つけられたのか？', 
    'つっつけようったって．．．', 
    'こいつは眼が．．．', 
    '匂いですよ', 
    'レーネさんが作ったハンバーグの美味しそうな香り', 
    'それを辿ってここまで来てみたら．．．'
    ]

e_023 = [
    '腐肉を煮詰めたような臭いが充満していて', 
    '大変気分が悪い', 
    'このままではせっかくの夕飯も冷めてしまう', 
    'お互い銃を抜くことなく',
    'どうか穏便に帰して頂けませんか', 
    'スッ．．．', 
    'ビル', 
    'おう', 
    '状況が全く見えてないマヌケには', 
    '鉛弾で思い知らせてやらねばな', 
    '閉じ眼如きが舐めた口ききやがって', 
    'キャッ', 'シオラッ！！'
    ]

e_pages = [e_020, e_021, e_022, e_023]
o_pages = [
    oma_text_isolator(ts, 20),
    oma_text_isolator(ts, 21),
    oma_text_isolator(ts, 22),
    oma_text_isolator(ts, 23)
]


def page_text_similarity(p1, p2):
    if len(p1) != len(p2):
        return "Length mismatch!"
    
    text_n = len(p1)
    similarities = []
    for i in np.arange(text_n):
        similarities.append(SequenceMatcher(None, p1[i], p2[i]).ratio())

    average = sum(similarities) / text_n
    
    return similarities, average
        
