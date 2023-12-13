##Overview

Amazon Kindle, a leading platform for digital reading, offers a vast collection of e-books accessible worldwide . The star ratings of e-books on Kindle are pivotal in guiding customer purchases, reflecting the quality and appeal of the content. Accurate prediction of these ratings can significantly aid Amazon in crafting targeted business and marketing strategies, tailoring recommendations to customer preferences, enhancing user satisfaction, and potentially boosting sales. This predictive analysis is key for Amazon to maintain its competitive edge in the dynamic digital book market.

##Data

The Amazon Kindle Book Dataset was sourced from Kaggle, and it was originally scrapped from publicly available data source in October 2023 [2]. It encompasses information on e-book publications available on the Amazon Kindle platform, accessible to any user. With the data collection occurring shortly before the commencement of this project, the dataset is current and reflects the latest trends.
The raw data contains 133,102 entries and 16 columns. Continuous target variable “stars” represents the ratings given to each e-book, with a possible score ranging from 1 to 5. During the data cleaning process, entries with missing target variables and columns unrelated to the regression questions were removed. The refined data has 129,920 data points and 11 features: 2 continuous features, 6 categorical features, and 3 time-based ordinal features.
Link to kaggle: https://www.kaggle.com/datasets/asaniczka/amazon-kindle-books-dataset-2023-130k-books/data
