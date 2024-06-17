**Stock News Interpreter**
With the recent stock market volatility, scepticism and pessimism of the stock market has been trending of late.[1] Additionally, new investors with no background in financial or economics education may be overwhelmed by the financial jargon and amount of information available online, driven by the gaining traction of financial bloggers and various independent news sources online.

Using this project, new investors will be able to have in their arsenal an application capable of performing topic modelling and logistic regression that will pick up cues on financial news and predict the market performance. Using that alongside financial indicators and trend analysis, investors would have an added edge in the market.

With logistic regression, a further understanding of the key or critical semantic or financial terminology and their impact on financial performance can be determined. This presents another opportunity for new investors to learn new financial concepts and to train them to be more independent, analytical investors on their own. In the context of learning, having extra resources would put the average investor in a better position against institutional investors.

**Features**
The news interpreter application was built using Latent Dirichlet Allocation as the topic modeller and VADER (Valence Aware Dictionary and sEntiment Reasoner) [1]. Outputs from the topic modeller and sentiment analysis are used as inputs to a logistic regression model for stock performance label generation.

To use this app, use either methods below:

[1] Clone project

[2] python -m venv venv

[3] .\venv\Scripts\

[4] pip install -r requirements.txt

[5] Run "applauncher.bat"

*Citation*
[1] Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
