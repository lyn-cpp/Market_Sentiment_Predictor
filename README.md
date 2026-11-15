# Market_Sentiment_Predictor
**Deep Multimodal Gated Two-Tower Cross-Attention Network (GRN + Macro-Guided News Attention + LightGBM + Logistic regression stacking)**

_This is not investment advice._

The user will input a news headline into our model, and the model will predict if 4 indexes: S&P 500, Dow Jones, Nikkei 225, and China SSE will go up, down, or stay neutral (i.e. within 0.35% fluctuation). Compare to the problem outlined in the guide, our model is trained on data about different indexes and markets, and considers the topic of each news headline to assign a different weight to how it will influence each market, because these markets behave differently. Furthermore, the “neutral” option added into our model accounts for scenarios where a news might not affect the stock. For example, a small volcanic eruption at Tonga Island won’t affect Dow Jones stocks very much. Similarly, a news that affects Dow Jones might not have such an effect on Nikkei 225. Our model attempts to incorporate these possibilities to come up with a more insightful prediction.

visit https://marketsentimentpredictortabreadme-ov-file-eamrvszbpcsiwgiwsjfv.streamlit.app
