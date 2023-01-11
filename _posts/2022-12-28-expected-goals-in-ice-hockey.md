---
title: "Building an expected goals model in ice hockey"
date: "2022-12-28"
classes: wide
layout: single
toc_left: yes
words_per_minute: 275
tags:
- data analysis
- classification
- python
- hockey
header:
  teaser: "/assets/images/hockey-shot.jpg"
---

One of the most popular metrics to evaluate shots is that of expected goals, which is commonplace in both soccer and ice hockey. Expected goals (xG) are computed using machine learning models, but what impacts the probability of a shot becoming a goal and how can we model this? These questions and more will be answered in this post. The code for this repository can be found [here](https://github.com/Safvenberger/NHL-xG).

Before moving further, let us first shed light on the first question that might arise. What exactly does **expected goals** refer to? Simply put, expected goals is a probability that describes how likely it is for a shot to become a goal. Note that a shot is considered in this context to have two outcomes: goal or no goal. Moreover, since it is a probability, it ranges from 0 to 1, where lower values indicate that it is not very probable that it becomes a goal and vice versa. 
{: .notice--success}
{: .text-justify}

## Data

To create a model for xG we also need some data. For this post, the data used are all shots with location data from the NHL between the 2010-2011 and 2021-2022 regular seasons, which was obtained from NHL's API. Each shot contains information regarding its outcome(goal/no goal), location (x and y coordinates), the type of shot (wrist/snap/slap/backhand/tip-in/wrap-around/deflected), as well as the player and team who took the shot. Here all penalty shots, including penalty shootouts were excluded. In total there were 1 170 271 shots of which 77 720 were goals. For the coordinate data recorded, the NHL rink has a length ($x$) of 200 feet and width ($y$) of 85 feet. The faceoff spot in the center circle is given by (0, 0), whereas the opposition's net is found at (89, 0) and the own net at (-89, 0). The negative $y$-coordinates correspond to the left wing and positive $y$-coordinates to the right wing from the perspective of the attacking team. For the defending team these coordinates are mirrored. Moreover, the rink is divided into three zones: defensive, neutral, and offensive.


## What impacts the probability of scoring?

As one might expect, two of the most prominent features in determining whether a goal is scored or not are distance and angle. Logically, the closer you are to the net, the more probable is it to score. Similarly, if you see more of the goal you have more options where to aim and thus outsmart the goalie.

How does this intuition translate to the data? To get some insight into this we will be performing an univariate[^1] visualization of some variables, where each has been round to the nearest integer. Let us start by checking distance:

<figure>
  <a href="https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/goal_proportion_Distance.png"><img src="https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/goal_proportion_Distance.png"></a>
  <figcaption>Probability of scoring by distance.</figcaption>
</figure>

At a first glance we see that the closer the shooter is to the opposition net, the likelier they are to score which is logical. The goal probability has a steady decrease until a distance of about 75, where it starts fluctuating more and having local peaks. From a theoretical point of view, the maximal distance attainable in the offensive zone is 77 feet [[1]](#1). This could imply that there is an increase in goal probability at the intersection of the neutral and offensive zone, which could be attributed to shots toward an empty net. Similarly, this is also likely to explain the slight increase seen as the distance ranges from 100 toward 200 as shots from the defensive zone are rare, with the exception of attempts toward an empty net. 

Based on this, we can assume that distance should be a variable in deciding if a shot is a goal or not[^2], but another variable that work in unison with distance is that of angle. We can also examine how the goal probability varies with angle.

<figure>
  <a href="https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/goal_proportion_Angle.png"><img src="https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/goal_proportion_Angle.png"></a>
  <figcaption>Probability of scoring by angle.</figcaption>
</figure>

Here we can see that goal probability varies with angle, but there are no clear optimal range of angles as was the case with distance. However, some angles have local peaks where the goal probability is higher than neighboring angles. Examples of this is angle of 0 and 45. Intuitively, a lower angle corresponds to a better view of the net, which means that they could be more potent for goalscoring, but there is not clear evidence for this in this univariate figure. 

Next, we can also consider three factors that also may affect the goal probability: goal differential,  manpower, and type of shot. 

<figure>
  <a href="https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/goal_proportion_manpower_GD.png"><img src="https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/goal_proportion_manpower_GD.png"></a>
  <figcaption>Probability of scoring for different manpower situations, goal differences, and shot types.</figcaption>
</figure>

From this figure we see that there is also evidence of both variables playing a part in scoring. For goal differential, the higher probabilities were seen while leading the game, which could once again be attributed to shooting toward an empty net, but also the fact that teams that have a lead may be less likely to force shots and instead aim for better quality shots. When it comes to manpower, the information portrayed show that even strength shots are the least probable to be a goal, while power play, with both one and a two-man advantage, are more probable and short-handed shots have the highest goal probability. Again, this agrees with the intuition that special teams, i.e. short-handed or power play, typically result in better shots, with respect to goal scoring probabilities. There is also the possibility that this may be a factor of less shots being taken in these situations compared to even strength. Regarding shot type we can also discern that there is indeed a difference in goal proportion per shot type. Based on the number of shots, tip-ins are the most probable to be goals, while slapshots are the least probable. Intuitively, this seems logical as slapshots are normally taken further from the net while tip-ins are very close to the net and shots from closer to the net tend to result in more goals.

Lastly, let us also turn our attention to some additional variables that could have predictive power for this case. For one, the elephant in the room thus far has been shots toward an empty net, which is worth investigating. Similarly, forwards typically have better shooting percentages than defensemen, while simultaneously being more likely to score when they are shooting from their off-wing.

![binary](https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/binary_variables.PNG){: width="50%" .align-center}

From this table, we note that shots by forwards tend to be lead to more goals. This should not come as much of a surprise as forwards are typically the players tasked with scoring. It also appears to be more beneficial for a player to shoot from their off-wing, likely due to better angles available. Yet the largest difference between goals and non-goals occur when shooting toward an empty net, a result that is to be expected logically as it either misses the net entirely or becomes a goal.

[^1]: Univariate means we are only considering one variable.
[^2]: Although the univariate results may indicate a relationship, this need not always hold for multivariate models. However, distance is typically one of the most important variables for expected goal models.

## Feature selection and engineering

Next let us move onward to the selection and engineering of the features that will be used to estimate the probability of scoring. In this context, a *feature* is a variable that describes some underlying factor in what affects the goal probability. Moreover, an *estimation* can be described as a well-qualified 'guess' based on data. 

### Distance and angle

As we saw in the previous section, distance and angle are two features that can impact the goal- scoring probability, and as a result we should include those for our estimation. 
The distance $d$ and angle $\gamma$ of a shot is calculated as: 

$$d(x, y) = \sqrt{(89 - x)^2 + y^2} \hspace{5mm} \gamma(x, y) = \tan^{-1}\left(\frac{y}{89-x}\right) \left(\frac{180}{\pi}\right)$$

where $(x, y)$ are the coordinates of the shot, where the $x$-coordinates at $x=89$ have been nudged to avoid numerical issues.

### Sequential features

To nuance the data in terms of play sequences, we will also consider what happened before a shot was taken. Note, however, that we still do not have access to everything that happened previously, such as passes, zone entries etc. But what we do have is a previous event, what team performed the event, and how long and far away it was from the current shot. In this post, we will not be considering explicit features for if a shot was a rebound or rush. Instead, we will use the same strategy as outlined by [Moneypuck](https://moneypuck.com/about.htm), that is we will use speed between events. In this context, speed is defined as the distance between two sequential events divided by the time between them. Additionally, for rebound shots we also compute the change in angle and again divide by the time difference to adjust for the different goal-scoring opportunitites that may arise for rebounds. 

### Auxilliary features

As has been noted on a multitude of occasions, there exist a bias in NHL stats. This bias exist for both counting stats, e.g. number of shots, hits and takeaways, as well as their coordinates [[3]](#3). There are many different proposals on how to deal with these biases, although they all share the same predicament: it is impossible to know if the corrections actually improve the data without over-correcting or if they introduce another bias. Consider, as an example, that shots at Madison Square Garden by the New York Rangers tend to be closer to the opposition's net than the away team. Although we may see this in the data, correcting for it may be both convoluted and inconsistent, as there is no universal method to apply for such a correction. Furthermore, the possibility of introducing a new systematic bias without knowing is something that should be considered prior to applying any correction of the actual data.

For this post, instead of correcting the actual data, we will instead add some auxiliary features to attempt to account for *some* of the inconsistencies in the data. More specifically, we add features to adjust for the location of the shot by considering the zone from which it was taken (defensive/neutral/offensive) and if it was taken behind the net or not. Here we are expecting shots from the defensive zone to be less threatening than an equivalent shot from the offensive zone. Additionally, we are also expecting the shots from behind the net, with a similar distance and angle, to lead to fewer goals than a shot from in front of the net.

**An additional note on data quality:**
From the data it is evident that mistakes are present across all seasons as some data do not make sense. For instance, there are tip-ins and wrap-arounds from the defending team's goal-line, as well shots from a (NHL reported) distance that do not align with the actual coordinates of the shot. Moreover, the distinction between a deflected shot and a tip-in can also vary, as two similar shots may be tagged differently. As a result, there will be a latent error in the data and consequently the xG estimations. 
{: .notice--danger}
{: .text-justify}

### Binary and dummy features

We have also previously seen that some binary features appear to impact goal-scoring.In addition, there are also a set of *dummy* features that may have an impact. A dummy feature is a numerical transformation of a categorical variable into a numerical one. However, to be used in the estimation we have to convert these variable to numerical versions. 

Consequently, the following features have been defined: 

| Feature             | Values                                                                    |
| :---                | :---                                                                      |
| IsForward           | 1 if forward, 0 otherwise.                                                |
| OffWing             | 1 if shot from off-wing, 0 otherwise.                                     |
| IsHome              | 1 if home team shot, 0 otherwise.                                         |
| BehindNet           | 1 if shot from behind the net, 0 otherwise.                               |
| ManpowerDummy       | EV, PP1, PP2, SH.                                                         |    
| ShotTypeDummy       | Slap, Snap, Wrist, Backhand, Tip-in, Backhand, Deflected.                 | 
| GoalDifferenceDummy | $\leq$-3, -2, -1, 0, 1, 2, $\geq$3.                                       |
| LastEventTeamDummy  | Same or different team for faceoffs, shot on goal, missed shot, and other.|
| ZoneDummy           | Defensive, neutral or offensive zone.                                     |

where the manpower is in terms of number of skaters for each team and PP1/PP2 denote how large the man advantage is. Furthermore, the goal difference is from the perspective of the shooting team as is the last event feature.

**A note on shooter and goaltender quality:**
In this post we are yet to discuss the topic of shooter quality and goaltending and how that may impact if a goal is scored or not. Naturally, a better shooter is more likely to score than a worse shooter and a better goaltender is more likely to concede less than a worse goaltender. However, the inclusion of such features in a machine learning model is tricky as they can be defined is many different ways as well as below-average shooters/goaltenders being penalized for having a career-year. Moreover, it is possible that human-imposed bias are unknowingly added if one is not careful in how they are encoded. Thus, for this particular model, we will not be including any specific features for who the shooter and goaltender is. Instead, we will continue the general case where the quality is entirely dependent on the shot itself.
{: .notice--warning}
{: .text-justify}

## The model

Now that we have all these features, we need to use them somehow to estimate the goal-scoring probability, i.e., xG. The xG model is an example of a *binary* classification task, where the outcome has two possible outcomes. In this case, the outcome $y$ is defined as

$$y = 
  \begin{cases} 
    1, \text{ if the shot is a goal} \\ 
    0, \text{ otherwise}
  \end{cases}
$$

and the task is to estimate a probability of each shot becoming a goal. This probability then becomes the xG value. Ultimately, these probabilities should accurately reflect the danger of the shot. That is, among goals that are all but certain, we wish to find the highest xG values. Examples of shots that should have a high xG are shots toward an empty net or rebound shots where the goalie is out of position. Similarly, shots that have no business being a goal should have a low xG. However, as the avid watcher of hockey would attest to, there is an inherent randomness to goals, as some goals arise from nothing. Consider for instance a deflected shot, an own goal, or a mistake by the goalie which all can lead to a goal despite not being a dangerous shot. 

Before moving on the the results, we first need to select a *model* to estimate these probabilities. In particular, we need to select a *supervised* model, which means that we have a target, here if a shot is a goal or not, that is known. 

A machine learning **model** refers to an algorithm that is used to uncover patterns and structure in data. Typically, a model is trained on a **training set**, which allows the model to learn the patterns that are present in the data. These patterns are then evaluated on a **test set**, which consists of data the model has not yet seen. Separating these two sets are important, as allowing the model to use information from the test set usually leads to **overfitting**. Overfitting means that the model learns the data **too** well and the generalization to new data becomes poor. In general, we want the model to also generalize to new data at a similar level to that of the training data. Lastly, it is common for a model to also have a set of **hyperparameters** that control the performance of model. As hyperparameters typically have a large range of possible values, we want to select the values that give optimal performance. For this purpose, we can also use a **validation set**, which allows the model to learn from the training set and evaluate on the validation set, while keeping the test set unused. 
{: .notice--primary}
{: .text-justify}

There exist many models to choose from, but one of the most popular is XGBoost [[2]](#2), which is also the choice of model here. Note that there is no association between the xG in expected goals and XGBoost.

In short, **XGBoost** is a decision tree ensemble that uses gradient boosting to learn the hyperparameters. Let's break that down. A *decision tree* is a method that builds a tree-based structure that is used for prediction. The tree consists of a set of splits, which are learned from the data and allows us to progress down from the root to the leaves where a prediction is made. An *ensemble* simply means that many decision trees are created and they "vote" together as a group (=ensemble). *Gradient boosting* refers to a method in which weak learners, i.e. a method that tends to have poor performance in isolation, are combined into one strong learner through iterative updating. That is, we start with a model that is typically not high-performing in and of itself, but can be used to create a better model.
{: .notice--primary}
{: .text-justify}

One explanation of the popularity for XGBoost is that the combination of minimizing training loss and regularization allows the resulting model to generalize well, but not too well. In addition, it is also computationally efficient. For the purpose of this post, XGBoost will be used to estimate the goal probability of shots, based on the features previously defined. For details on the hyperparameter optimization refer to the appendix.

## Results

Prior to training the model, we need to decide what data to use. Since we may expect that shots from different manpower situations differ from one another, the data was divided into four disjoint sets. The sets are even strength shots (EV), power play shots (PP), short-handed shots (SH), and empty net shots (EN). EV shots occur when there are an even amount of skaters for each team, PP shots when the shooting team has a numerical advantage with respect to skaters, SH shots are from a team with a numerical disadvantage, whereas EN shots are shots toward an empty net. Next, the data was split into a training, validation and test set for model learning and evaluation. The training set consisted of all seasons between 2010-2011 and 2019-2020, while the validation and test set had the 2020-2021 and 2021-2022 seasons, respectively. This was applied for each model. After training the models we want to evaluate how well the models performed, both with regards to what features are the most important as well as how well it generalizes to unseen data. 

First, let us examine the feature importance. That is, how much each feature contributes to the estimation of a goal where a higher value means it is more important.

<figure class="half">
    <a href="https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/feature_importance_EV.png"><img src="https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/feature_importance_EV.png"></a>
    <a href="https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/feature_importance_PP.png"><img src="https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/feature_importance_PP.png"></a>
    <figcaption>Feature importance for even strength shots (left) and power play shots (right).</figcaption>
</figure>

<figure class="half">
    <a href="https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/feature_importance_SH.png"><img src="https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/feature_importance_SH.png"></a>
    <a href="https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/feature_importance_EN.png"><img src="https://raw.githubusercontent.com/Safvenberger/NHL-xG/main/Figures/feature_importance_EN.png"></a>
    <figcaption>Feature importance for short-handed shots (left) and empty net shots (right).</figcaption>
</figure>

The first thing to note is that shot distance is the most important feature across all four models. We can also see that the type of shot is also a prominent factor in estimating a goal, but the type of shot varies depending on the state of play. For instance, in PP slapshots are the most important, followed by tip-ins. Meanwhile in EV wrap-arounds and backhands are the shots that best separate a goal and no-goal. The angle and angle change are also features that impact the probability of scoring, while speed and distance from the last event are also important. We can also observe that the "correction" features, i.e. Zone and and BehindNet, also help in separating a goal from a no-goal. Lastly, it is noteworthy that each model has a set of features which are not important in the estimation, e.g. PP2 for PP shots and DefZone for EV shots. 

Another aspect of model evaluation is to examine some evaluation metrics; in this case we will look at log loss and the area under the receiver operating characteristic (ROC) curve (AUC). The log loss is a loss based metric where we want as low a value as possible, whereas for the AUC we want a value as close to 1 as possible. For the log loss, the larger the distance between the target (1 or 0) and prediction probability (between 0 and 1) the larger the loss. On the other hand, AUC considers how well separable and distinct the target classes are. For this data, this means how well we can recognize goals that are goals and no-goals that are no-goals. When this distinction between the classes becomes clearer the AUC value increases. However, before considering how 'well' the models performed, we need to define what 'good' performance means as the metrics are relative to some baseline. There are a variety of baselines that can be used, such as comparing the same metric of two machine learning models or using a naive baseline classifier. Here, we will be using a baseline 'dummy' classifier that always predicts the majority class, here no-goal as they are more common than goals. The following table summarizes the results of the baseline and the metric for each model:

<table>
  <colgroup>
    <col width="20%" />
    <col width="20%" />
    <col width="20%" />
    <col width="20%" />
    <col width="20%" />
  </colgroup>
  <thead>
    <tr class="header">
    <th style="text-align: center">Model</th>
    <th style="text-align: center">Baseline AUC</th>
    <th style="text-align: center">Model AUC</th>
    <th style="text-align: center">Baseline log loss</th>
    <th style="text-align: center">Log loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td markdown="span" style="text-align: center">EV</td>
      <td markdown="span" style="text-align: center">0.5</td>
      <td markdown="span" style="text-align: center">0.772</td>
      <td markdown="span" style="text-align: center">2.163</td>
      <td markdown="span" style="text-align: center">0.205</td>
    </tr>
    <tr>
      <td markdown="span" style="text-align: center">PP</td>
      <td markdown="span" style="text-align: center">0.5</td>
      <td markdown="span" style="text-align: center">0.692</td>
      <td markdown="span" style="text-align: center">3.343</td>
      <td markdown="span" style="text-align: center">0.296</td>
    </tr>
    <tr>
      <td markdown="span" style="text-align: center">SH</td>
      <td markdown="span" style="text-align: center">0.5</td>
      <td markdown="span" style="text-align: center">0.827</td>
      <td markdown="span" style="text-align: center">2.362</td>
      <td markdown="span" style="text-align: center">0.201</td>
    </tr>
    <tr>
      <td markdown="span" style="text-align: center">EN</td>
      <td markdown="span" style="text-align: center">0.5</td>
      <td markdown="span" style="text-align: center">0.689</td>
      <td markdown="span" style="text-align: center">11.91</td>
      <td markdown="span" style="text-align: center">0.575</td>
    </tr>
  </tbody>
</table>

From the table we note that, unsurprisingly, our models outperforms the baseline classifier across all cases. The best performance was found for the SH model, followed by EV, while PP and EN had the lowest performance. Here we can also believe that indeed there is a difference in how the game is played, and how goals are scored, in different game situations. Overall, we seem to be able to capture some of the variation in scoring goals, but all the variation can not be found, and neither should we expect to since goals have an inherent randomness to them. 

Finally, let us now turn to the topic of player evaluation. More specifically, the scoring ability of players from the 2021-2022 NHL season based on the difference between actual goals scored and expected goals scored. A positive value indicates that a player scored more than they were expected to whereas a negative value means they scored less than expected. 

Why should we consider the Goals-xG instead of just xG? Well, using the difference we can actually compare the actual and expected scoring of a player, to determine the best scorers in the league. As an example, let us compare Connor McDavid and Auston Matthews from the last season. Connor McDavid had an xG of 45.53, which was highest in the league, while Auston Matthews was second with 40.76 xG. However, McDavid scored 44 goals while Matthews scored 60. Consequently, despite being expected to score the most in the league, McDavid was outscored by 16 goals compared to Matthews. 
{: .notice--success}
{: .text-justify}

<iframe src="../assets/html/xgHockey.html" height="750px" width="100%" style="border:none"></iframe>

As the interactive table highlights, the players who were the best finishers were Auston Matthews (60 goals vs. 40.76 xG), Kirill Kaprizov (47 goals vs. 29.38 xG), and Chris Kreider (52 goals vs. 35.70 xG). These results make sense as Matthews was the league's leading goalscorer and won the Hart trophy (Most valuable player in the league), while Kaprizov set a new franchise record for most goals scored in a single season, and Kreider had a career year, goal-scoring wise, having nearly dubbled his previous season-high of 28 goals. Overall, the high ranking players all have one thing in common: they are good at scoring goals. Similarly, Kaprizov and Kreider both were both lethal during power play situations while Matthews did most of his scoring in even strength. 

On the other hand, among the most inefficient scorers we find Brendan Gallagher (7 goals vs. 19.62 xG), Alexander Radulov (4 goals vs. 13.64 xG), and Phil Kessel (8 goals vs 17.63 xG). These players have likely all passed their peak, and all recorded career-lows of number of goals scored during a single season. Again, this is logical as none of these players had a particularly good season last year. 

**Future considerations**: In conclusion, we have created an expected goals model that quantifies the how many goals each player is expected to score. Overall, the models have an AUC around 0.7 to 0.8, which is a reasonable value according to what is achievable using the publicly available data, whereas commercial xG models may reach somewhat higher AUC values [[4]](#4). Natural extensions of the model used here would be to include features regarding e.g. passing and zone entries as the can be tracked in event data. Information from tracking data, such as the speed, balance, and direction of the shooting player may also improve the model, as well as the positioning of the other players on the ice. Additionally, the work done by Micah Blake McCurdy [[5]](#5) to incorporate shooting/goalie talent can also help differentiate between the quality of shooting and goaltending. 
{: .notice--primary}
{: .text-justify}

## Acknowledgements
All data with player stats are from the NHL, while the team badges, which are the property of the NHL and its teams, were retrieved from ESPN. This post also drew inspiration from the work by Josh and Luke Younggren (EvolvingWild), Peter Tanner (MoneyPuck), and Micah Blake McCurdy (HockeyViz).

## Appendix

### Hyperparameter optimization
In XGBoost the task is to find the parameters $\theta$ that provides the provide the best fit between the features $x_i$ and the targets $y_i$, which is done by optimizing an objective function of the form

$$\text{obj}(\theta) = L(\theta) + \Omega(\theta)$$

with training loss $L$ and $\Omega$ controlling the regularization. In this post we have used the log loss as the objective function, that is:

$$L(\theta) = \sum_{i} \left[y_i \log(\Pr(y=1)) + (1-y_i) \log(\Pr(y=0)) \right]$$

where $y=1$ and $y=0$ correspond to a goal and no-goal, respectively. To optimize $\theta$ the package `hyperopt` was used to conduct Bayesian optimization. For this we need three things: a search space, an objective function, and a minimizer. First, let us define a search space: 

```python
# Define the search space for optimization
space = {"max_depth":        hp.quniform("max_depth", 2, 10, 1),
         "subsample":        hp.quniform("subsample", 0.5, 0.9, 0.05),
         "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),
         "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
         "max_delta_step":   hp.quniform("max_delta_step", 0, 10, 1),
         "learning_rate":    hp.quniform("learning_rate", 0.001, 0.5, 0.05),
         "gamma":            hp.qloguniform("gamma",  -10, 5, 1),
         "alpha":            hp.qloguniform("alpha",  -10, 5, 1),
         "lambda":           hp.qloguniform("lambda", -10, 5, 1),
         # Non-tuned components
         "n_estimators": 100, 
         "random_state": 0, 
         "seed": 0}
```

and an objective function where the objective should be minimized: 

```python 
def objective(space: Dict, X_train: DataFrame, y_train: DataFrame, 
              X_val: DataFrame, y_val: DataFrame) -> Dict:
    """
    Define the objective (minimization) function used for optimization.
    Parameters
    ----------
    space : Dict
        The parameter search space.
    X_train : DataFrame
        Design matrix to use for training.
    y_train : DataFrame
        Labels to use for training.
    X_val : DataFrame
        Design matrix to use for validation.
    y_val : DataFrame
        Labels to use for validation.
    Returns
    -------
    Dict
        Dictionary containing the current loss and the status of optimization.
    """
    
    # Define the classifier and its parameters
    xg_clf = xgb.XGBClassifier(max_depth=int(space["max_depth"]), 
                               gamma=space["gamma"],
                               subsample=space["subsample"],
                               min_child_weight=int(space["min_child_weight"]),
                               max_delta_step=int(space["max_delta_step"]),
                               learning_rate=space["learning_rate"],
                               reg_alpha=space["alpha"],
                               reg_lambda=space["lambda"],
                               n_estimators=space["n_estimators"],
                               random_state=space["random_state"],
                               seed=space["seed"],
                               use_label_encoder=False, 
                               objective="binary:logistic",
                               eval_metric="logloss")
    
    # Fit the classifier on the training data
    xg_clf.fit(X_train, y_train, verbose=False)
    
    # Predict the probability of scoring on the validation set
    y_hat = xg_clf.predict_proba(X_val)
    
    # Compute the log loss of the predictions
    loss = log_loss(y_val, y_hat[:, 1])
    
    return {"loss": loss, "status": STATUS_OK }
```

and lastly, we define a function to retrieve the best hyperparameters after minimizing the objective function:

```python
def optimize_model(X_train: DataFrame, y_train: DataFrame, 
                   X_val: DataFrame, y_val: DataFrame, 
                   space: Dict, max_evals: int=100) -> Dict:
    """
    Optimize a model with a given objective function and parameter space.
    Parameters
    ----------
    X_train : DataFrame
        Design matrix to use for training.
    y_train : DataFrame
        Labels to use for training.
    X_val : DataFrame
        Design matrix to use for validation.
    y_val : DataFrame
        Labels to use for validation.
    space : Dict
        The parameter search space.
    max_evals : int, optional. Default is 100.
        The maximum number of evaluations to optimize for.
    Returns
    -------
    best_hyperparams : Dict
        Dictionary containing the hyperparameters that minimize the objective function.
    """
    
    # Storage for optimization results
    trials = Trials()

    # Perform function minimization on a given search space and objective function
    best_hyperparams = fmin(fn=lambda x: objective(x, X_train, y_train, X_val, y_val),
                            space=space,
                            algo=tpe.suggest,
                            max_evals=max_evals,
                            trials=trials,
                            rstate=np.random.default_rng(0))
   
    return best_hyperparams
```

That is it! Now we have gotten our optimal parameters for a given model and we can now use those parameters in fitting the final model.

## References
<a id="1">[1]</a> 
NHL Shot Quality 2009-10, Ken Krzywicki
[Link](http://hockeyanalytics.com/Research_files/SQ-RS0910-Krzywicki.pdf)

<a id="2">[2]</a> 
Machine Learning Challenge Winning Solutions, Github.
[Link](https://github.com/dmlc/xgboost/tree/master/demo#machine-learning-challenge-winning-solutions)

<a id="3">[3]</a> 
Total Hockey Rating (THoR): A comprehensive statistical rating of National Hockey League forwards and defensemen based upon all on-ice events
M Schuckers, J Curro - 7th annual MIT sloan sports analytics conference, 2013
[Link](https://www.statsportsconsulting.com/wp-content/uploads/Schuckers_Curro_MIT_Sloan_THoR.pdf)

<a id="4">[4]</a> 
Expected Goals (xG) Models Explained, JetsNation HQ
[Link](https://jetsnation.ca/news/expected-goals-xg-models-explained)

<a id="5">[5]</a> 
Magnus 6: xG, Shooting, and Goalie-ing, Micah Blake McCurdy
[Link](https://hockeyviz.com/txt/xg6)

