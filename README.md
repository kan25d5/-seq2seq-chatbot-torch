# seq2seq-chatbot-torch
## Task
Sequence to Sequence model that learns responses to utterances with twitter corpus.

## Dataset
The contents of the corpus json file

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