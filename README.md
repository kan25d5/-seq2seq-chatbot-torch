# seq2seq-chatbot-torch
## Task
Sequence to Sequence model that learns responses to utterances with twitter corpus.

## Dataset
The contents of the corpus json file.

```json
[
    // dialogue set
    [
        // dialogue(Same tweet thread)
        [
            // Chain of tweets (messages-response)
            {
                // message tweet
                "text" : "aaaaaaaaaaaaaaaaaaaaa!!!",
                "screen_name" : "@AAAAAAAAA",
                "tweet_id" : 10934893023490
            },
            {
                // responce tweet
                "text" : "bbbbbbbbbbbbbbbbbbbbb!!!",
                "screen_name" : "@BBBBBBBB",
                "tweet_id" : 10934893023450
            },
            {
                // Response to "responce tweet"
                "text" : "ccccccccccccccccccccc!!!",
                "screen_name" : "@AAAAAAAAA",
                "tweet_id" : 10934893023423
            }
            // .....
        ]
        // ....
    ]
]
```

## Using library
- pytorch==1.11.0 
- pytorch-lightning==1.6.0 
- fugashi==1.1.2
- tensorflow==2.4.1
- gensim==4.1.2
- scikit-learn==1.0.2
- dill==0.3.4

## Models
 - Encoder-Decoder
 - Encoder-Decoder + Attention
 - Transformer (main model)

## Example of response prediction
- dataset dialogue paris length
  - train_dataset : 446463
  - val_dataset : 137047
  - test_dataset : 59010
- BATCH SIZE : 100
- EPOCH SIZE : 98
- tag description
  - source : Message during the dialogue corpus (Source sequence).
  - target : Response to message during the dialogue corpus  (Target sequence).
  - preds : Response to the model-generated Source (Greedy Decoding).


```
dataloader_type : train
current epoch : 98
source : <s>まじ?それは嬉しい( ♡ 艸 ♡ )</s>
target : <s>沢山お話しようね!!楽しみ!ฅ^•ω•^ฅ</s>
preds : <s>うん!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
--------------------------
source : <s>沢山お話しようね!!楽しみ!ฅ^•ω•^ฅ</s>
target : <s>今回って事前予約みたいなやつあるのかな...?</s>
preds : <s>楽しみにしてるね!!!</s>
--------------------------
source : <s>今回って事前予約みたいなやつあるのかな...? </s>
target : <s>ありそうだよね、、お金貯めなきゃ、、←</s>
preds : <s>そうだよね、、、、、、、、、私のとこにはないけど、、、、、</s>
--------------------------
source : <s>ありそうだよね、、お金貯めなきゃ、、← </s>
target : <s>バイトが3月(卒業)で解禁されるからそしたら頑張って貯めないと(́;ω;`)</s>
preds : <s>バイトしてなんかバイトしてたんだよね()</s>
--------------------------
source : <s>バイトが3月(卒業)で解禁されるからそしたら頑張って貯めないと(́;ω;`) </s>
target : <s>え、学校バイト禁止な感じ??そしたら<unk>とかじゃなきゃライブまでに給料日間に合わないよね...</s>
preds : <s>バイト禁止な</s>
--------------------------
source : <s>え、学校バイト禁止な感じ??そしたら<unk>とかじゃなきゃライブまでに給料日間に合わないよね...</s>
target : <s>バイト禁止(́;ω;`)なんならsnsも禁止←そうなの日雇い頑張らないと名古屋までの往復代とホテル代がないw</s>
preds : <s>バイト禁止なの???学校のためにバイト禁止なの??????????????????????バイトしてるんだけどバイトとかでバイトしてるんだよね???????</s>
--------------------------
source : <s>バイト禁止(́;ω;`)なんならsnsも禁止←そうなの日雇い頑張らないと名古屋までの往復代とホテル代がないw</s>
target : <s>いやあああキツ、、、なんだsns禁止てwwww結構するもんね、、w<unk>バイト運送会社はやめときな...アザだらけになるし腱鞘炎にもなりかけるからおすすめしない...</s>
preds : <s><unk>ぞ(́・ω・`)</s>
--------------------------
source : <s>そのうち『気持ちがヴィーガンなら肉を食べても良い』とか言い出して自分の都合の良いようにヴィーガンの基準を下げていきそう</s>
target : <s>動物からの搾取を行わないねぇ...。ヴィーガンが食べている野菜を収穫するまでにどれだけの生物を殺していると思っているんだろう?</s>
preds : <s>そうですね...........................................................
--------------------------
source : <s>動物からの搾取を行わないねぇ...。ヴィーガンが食べている野菜を収穫するまでにどれだけの生物を殺していると思っているんだろう?</s>
target : <s>ゼロだと思い込んでいるのかもしれませんね。肉を食わないだけで自分たちの手は汚れていないと思い込んでるような連中ですから。</s>
preds : <s>食べないと駄目なの...(笑)</s>
--------------------------
source : <s>いただきます</s>
target : <s>豪華やなぁ(*^○^*)</s>
preds : <s>いただきます!(=ヮ=*)૭</s>
--------------------------
source : <s>豪華やなぁ(*^○^*)</s>
target : <s>マックスバリュ鷲塚の惣菜とかです。美味しかったですよ</s>
preds : <s>美味しかったです(*́艸`)</s>
--------------------------

=================================

dataloader_type : test
current epoch : 98
source : <s>左手の手首がここまでしか上がらない......全然力入んないし、指伸ばそうと思ってもプルプルしちゃう......(ᴗ̩̩ᄉᴗ̩̩)いつ治るかな??</s>
target : <s>腕まくらしてた?</s>
preds : <s>私の場合、<unk>の足の指に指がついていかないといけないのよ.......</s>
--------------------------
source : <s>腕まくらしてた?</s>
target : <s>自分に腕枕してました</s>
preds : <s>いや、まだしてないよw</s>
--------------------------
source :<s><unk>の体の値段頭部:20円胸部:4000円腕:8円腰:400円股関:0円お尻:0円脚部:1円ちょwww全然価値ないやんけ。・゚・(ノд`)・゚・。</s>
target : <s>(*́・ω・)ノ(・ω・`*)ヨシヨシこれ......足??ラッキーの体の値段頭部:80円胸部:0円腕:<unk>万円腰:<unk>万円股関:10円お尻:0円<unk>:150億円</s>
preds : <s><unk>は1万円で<unk>円(`・ω・́)キリッ</s>
--------------------------
source : <s>(*́・ω・)ノ(・ω・`*)ヨシヨシこれ......足??ラッキーの体の値段頭部:80円胸部:0円腕:<unk>万円腰:<unk>万円股関:10円お尻:0円脚部:150億円</s>
target : <s>足やねwww何でやぁ(∩◉_◉∩);:</s>
preds : <s>私は10万くらいでいいよ。</s>
--------------------------
source : <s>足やねwww何でやぁ(∩◉_◉∩);:</s>
target : <s>足は意味が分からないwww腕枕をします(゚д゚)<</s>
preds : <s>それなww私足の指の指が痛くなるからさww</s>
--------------------------
source : <s>足は意味が分からないwww腕枕をします( ゚ д ゚ )</s>
target : <s>腕枕も高いわ(∩◉_◉∩);:</s>
preds : <s>私は足の裏で足が痛くて腕が痛くて途中で寝てました(;́д`)ハァハァ</s>
--------------------------
source : <s>腕枕も高いわ(∩◉_◉∩);: </s>
target : <s>筋肉落ちてきたから硬くない腕まくら(✧д✧)カッ!!</s>
preds : <s>私もそれ思ったわ</s>
--------------------------
source : <s>狭いところを手のひらぶらぶらしながら斜め後ろくらいに向けて歩いてたら、偶然ハーフパンツの部活帰りjkがヒップアタックしてきてガチでがっつり触ってしまったんですけど、これ痴漢になりますか(震え声)</s>
target : <s>うらやましい</s>
preds : <s>え、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、
--------------------------
source : <s>うらやましい</s>
target : <s>警察だ!‍</s>
preds : <s>今日も暑いから、早く行きたい(*́ω`*)</s>
--------------------------
source : <s>警察だ!‍</s>
target : <s>かわいかった?</s>
preds : <s>付けてください!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
--------------------------
source : <s>かわいかった? </s>
target : <s>ふつうくらいだったはず</s>
preds : <s>はい!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
--------------------------
```