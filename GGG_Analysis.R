library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)

# Read in the data
setwd("/Users/student/Desktop/STAT348/GGG")
#GGG_training_missing  <- vroom("/Users/student/Desktop/STAT348/GGG/trainWithMissingValues.csv")
GGG_training <-vroom("/Users/student/Desktop/STAT348/GGG/train.csv")

# Bake the recipe
GGG_recipe <- recipe(type~., data=GGG_training_missing) %>%
  step_impute_knn(bone_length, impute_with =imp_vars(has_soul), neighbors = 7) %>%
  step_impute_mean(rotting_flesh) %>%
  step_impute_median(hair_length)
  
prep <- prep(GGG_recipe)
baked_missing <-bake(prep, new_data=GGG_training_missing)

# RMSE
rmse_vec(GGG_training[is.na(GGG_training_missing)], baked_missing[is.na(GGG_training_missing)])

# KNN ---------------------------------------------------------------------
# Read in the data
setwd("/Users/student/Desktop/STAT348/GGG")
GGG_training <-vroom("/Users/student/Desktop/STAT348/GGG/train.csv") %>%
  mutate(type=factor(type))
GGG_test <-vroom("/Users/student/Desktop/STAT348/GGG/test.csv")

# Bake the recipe
GGG_recipe <- recipe(type~., data=GGG_training) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_lencode_glm(all_nominal_predictors(), outcome=vars(type))

prep <- prep(GGG_recipe)
bake(prep, new_data=GGG_training)
bake(prep, new_data=GGG_test)

# KNN setup(model and workflow)
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(GGG_recipe) %>%
  add_model(knn_model)

# Set up tuning grid and folds
folds <- vfold_cv(GGG_training, v = 5, repeats=1)

knn_tuning_grid <- grid_regular(neighbors(),
                                levels = 10)

# Tune neighbors here
knn_cv <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=knn_tuning_grid,
            metrics=metric_set(accuracy))

# Find the best Fit
bestTune <- knn_cv %>%
  select_best("accuracy")

# Finalize workflow and predict
final_wf <-
  knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGG_training)

final_wf %>%
  predict(new_data = GGG_test)

# Predict
GGG_predictions <- predict(final_wf,
                              new_data=GGG_test,
                              type="class") %>%
  bind_cols(GGG_test) %>%
  rename(type=.pred_class) %>%
  select(id, type)

vroom_write(x = GGG_predictions, file = "KNN.csv", delim = ",")


# Neural Network ----------------------------------------------------------
# Read in the data
setwd("/Users/student/Desktop/STAT348/GGG")
GGG_training <-vroom("/Users/student/Desktop/STAT348/GGG/train.csv") %>%
  mutate(type=factor(type))
GGG_test <-vroom("/Users/student/Desktop/STAT348/GGG/test.csv")

# Bake the recipe
nn_recipe <- recipe(formula=type~., data=GGG_training) %>%
  update_role(id, new_role="id") %>%
  step_normalize(all_numeric_predictors()) %>%
  step_lencode_glm(all_nominal_predictors(), outcome=vars(type)) %>%
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

prep <- prep(nn_recipe)
bake(prep, new_data=GGG_training)
bake(prep, new_data=GGG_test)

# Set up a model and workflow
nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%
            set_engine("nnet") %>%
            set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

# Set up a tuning grid and folds
nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 20)),
                            levels=10)
folds <- vfold_cv(GGG_training, v = 5, repeats=1)

# Tune it
nn_cv <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=nn_tuneGrid,
            metrics=metric_set(accuracy))

# Find the best Fit
bestTune <- nn_cv %>%
  select_best("accuracy")

# Finalize workflow and predict
final_wf <-
  nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGG_training)

final_wf %>%
  predict(new_data = GGG_test)

# Predict
GGG_predictions <- predict(final_wf,
                           new_data=GGG_test,
                           type="class") %>%
  bind_cols(GGG_test) %>%
  rename(type=.pred_class) %>%
  select(id, type)

vroom_write(x = GGG_predictions, file = "NeuralNetwork.csv", delim = ",")


graph <- nn_cv %>% 
  collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line() 
  ggsave(filename = "NNplot.png", plot = graph, width = 6, height = 4)


# Boosting ----------------------------------------------------------------
library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(bonsai)
library(lightgbm)

#Read in the data
setwd("/Users/student/Desktop/STAT348/GGG")
GGG_training <-vroom("/Users/student/Desktop/STAT348/GGG/train.csv") %>%
  mutate(type=factor(type))
GGG_test <-vroom("/Users/student/Desktop/STAT348/GGG/test.csv")

#Bake the recipe
GGG_recipe <- recipe(type~., data=GGG_training) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_lencode_glm(all_nominal_predictors(), outcome=vars(type))

prep <- prep(GGG_recipe)
bake(prep, new_data=GGG_training)
bake(prep, new_data=GGG_test)

#Set up a model and workflow
#model
boost_model <- boost_tree(tree_depth=tune(),
                            trees=tune(),
                            learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")
  
#workflow
boost_wf <- workflow() %>%
    add_recipe(GGG_recipe) %>%
    add_model(boost_model)



# Set up a tuning grid and folds
boost_tuneGrid <- grid_regular(tree_depth(),
                               trees(),
                               learn_rate(),
                              levels=10)
folds <- vfold_cv(GGG_training, v = 5, repeats=1)

# Tune it
boost_cv <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=boost_tuneGrid,
            metrics=metric_set(accuracy))

# Find the best Fit
bestTune <- boost_cv %>%
  select_best("accuracy")

# Finalize workflow and predict
final_wf <-
  boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGG_training)

final_wf %>%
  predict(new_data = GGG_test)

# Predict
boost_predictions <- predict(final_wf,
                           new_data=GGG_test,
                           type="class") %>%
  bind_cols(GGG_test) %>%
  rename(type=.pred_class) %>%
  select(id, type)

vroom_write(x = boost_predictions, file = "Boosting.csv", delim = ",")


# Bart --------------------------------------------------------------------
bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(GGG_recipe) %>%
  add_model(bart_model)

# Set up a tuning grid and folds
bart_tuneGrid <- grid_regular(trees(),
                               levels=10)
folds <- vfold_cv(GGG_training, v = 5, repeats=1)

# Tune it
bart_cv <- bart_wf %>%
  tune_grid(resamples=folds,
            grid=bart_tuneGrid,
            metrics=metric_set(accuracy))

# Find the best Fit
bestTune <- bart_cv %>%
  select_best("accuracy")

# Finalize workflow and predict
final_wf <-
  bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGG_training)

final_wf %>%
  predict(new_data = GGG_test)

# Predict
boost_predictions <- predict(final_wf,
                             new_data=GGG_test,
                             type="class") %>%
  bind_cols(GGG_test) %>%
  rename(type=.pred_class) %>%
  select(id, type)

vroom_write(x = boost_predictions, file = "Bart.csv", delim = ",")


# Last Submission Using Naive Bayes ---------------------------------------
library(tidymodels)
library(embed)
library(vroom)
library(discrim)

#Read in the data
setwd("/Users/student/Desktop/STAT348/GGG")
GGG_training <-vroom("/Users/student/Desktop/STAT348/GGG/train.csv") %>%
  mutate(type=factor(type))
GGG_test <-vroom("/Users/student/Desktop/STAT348/GGG/test.csv")

#Bake the recipe
GGG_recipe <- recipe(type~., data=GGG_training) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_lencode_glm(all_nominal_predictors(), outcome=vars(type))

prep <- prep(GGG_recipe)
bake(prep, new_data=GGG_training)
bake(prep, new_data=GGG_test)

## create a workflow with model & recipe
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
  add_recipe(GGG_recipe) %>%
  add_model(nb_model)

# set up tuning grid and folds

folds <- vfold_cv(GGG_training, v = 5, repeats=1)

nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness())

## Tune smoothness and Laplace here
nb_cv <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=nb_tuning_grid,
            metrics=metric_set(accuracy))

## Find the best Fit
bestTune <- nb_cv %>%
  select_best("accuracy")

# Finalize workflow and predict
final_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGG_training)

final_wf %>%
  predict(new_data = GGG_test)

## Predict

GGG_predictions <- predict(final_wf,
                           new_data=GGG_test,
                           type="class") %>%
  bind_cols(GGG_test) %>%
  rename(type=.pred_class) %>%
  select(id, type)

vroom_write(x = GGG_predictions, file = "NaiveBayes.csv", delim = ",")


