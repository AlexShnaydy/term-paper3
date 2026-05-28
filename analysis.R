# ============================================================================
# ЭТАП 1: ЗАГРУЗКА БИБЛИОТЕК И ДАННЫХ
# ============================================================================

library(dplyr)
library(lme4)
library(lmerTest)
library(car)
library(ggplot2)
library(patchwork)

# Загрузка данных
R_sentences <- read.csv("R_sentences.csv", stringsAsFactors = FALSE)
R_human <- read.csv("R_human.csv", stringsAsFactors = FALSE)
R_model <- read.csv("R_model.csv", stringsAsFactors = FALSE)

# ============================================================================
# ЭТАП 2: ПРЕОБРАЗОВАНИЕ ТИПОВ ДАННЫХ
# ============================================================================

# Данные о предложениях
R_sentences$sentence <- as.integer(R_sentences$sentence)
R_sentences$LogDice_wrong <- as.numeric(gsub(",", ".", R_sentences$LogDice_wrong))
R_sentences$LogDice_right <- as.numeric(gsub(",", ".", R_sentences$LogDice_right))
R_sentences$semantics <- as.numeric(gsub(",", ".", R_sentences$semantics))
R_sentences$pos <- as.factor(R_sentences$pos)
R_sentences$paronyms <- as.factor(R_sentences$paronyms)

# Данные людей
R_human$participant <- as.factor(R_human$participant)
R_human$sentence <- as.integer(R_human$sentence)
R_human$form <- as.factor(R_human$form)
R_human$order <- as.integer(R_human$order)
R_human$visible <- as.integer(R_human$visible)
R_human$acceptable <- as.integer(R_human$acceptable)
R_human$correctable <- as.integer(R_human$correctable)

# Данные модели
R_model$sentence <- as.integer(R_model$sentence)
R_model$temperature <- as.numeric(R_model$temperature)
R_model$attempt <- as.integer(R_model$attempt)
R_model$visible <- as.integer(R_model$visible)
R_model$acceptable <- as.integer(R_model$acceptable)
R_model$correctable <- as.integer(R_model$correctable)

# ============================================================================
# ЭТАП 3: ОБЪЕДИНЕНИЕ И ПОДГОТОВКА ДАННЫХ ЛЮДЕЙ
# ============================================================================

human_joined <- R_human %>% left_join(R_sentences, by = "sentence")

human_joined$participant <- as.factor(human_joined$participant)
human_joined$form <- as.factor(human_joined$form)
human_joined$sentence <- as.factor(human_joined$sentence)
human_joined$education <- factor(human_joined$education, 
                                 levels = c("высшее", "неполное высшее", "среднее"))
human_joined$pos <- as.factor(human_joined$pos)
human_joined$paronyms <- as.factor(human_joined$paronyms)
human_joined$last_form <- grepl("^Last", human_joined$form)

# Центрирование и масштабирование непрерывных предикторов
human_joined$age_c <- scale(human_joined$age, center = TRUE, scale = TRUE)
human_joined$LogDice_c <- scale(human_joined$LogDice_right, center = TRUE, scale = TRUE)
human_joined$semantics_c <- scale(human_joined$semantics, center = TRUE, scale = TRUE)
human_joined$order_c <- scale(human_joined$order, center = TRUE, scale = TRUE)

# Создание бинарной переменной для дополнительных правок (гиперкоррекция)
human_joined$additional <- ifelse(human_joined$add_inf != "" & !is.na(human_joined$add_inf), 1, 0)

# ============================================================================
# ЭТАП 4: РЕГРЕССИОННЫЙ АНАЛИЗ ДАННЫХ ЛЮДЕЙ
# ============================================================================

# --- 4.1. Влияние метаданных (возраст, образование) ---

# Замечаемость (visible)
model_vis_meta <- glmer(visible ~ age_c + education + 
                          (1|participant) + (1|form) + (1|sentence),
                        data = human_joined, family = binomial,
                        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
summary(model_vis_meta)
Anova(model_vis_meta)

# Приемлемость (acceptable) — только на замеченных
human_vis <- human_joined[human_joined$visible == 1, ]
model_acc_meta <- glmer(acceptable ~ age_c + education + 
                          (1|participant) + (1|form) + (1|sentence),
                        data = human_vis, family = binomial,
                        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
summary(model_acc_meta)
Anova(model_acc_meta)

# Точность попадания (correctable) — только на приемлемых
human_acc <- human_vis[human_vis$acceptable == 1, ]
model_corr_meta <- glmer(correctable ~ age_c + education + 
                           (1|participant) + (1|form) + (1|sentence),
                         data = human_acc, family = binomial,
                         control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
summary(model_corr_meta)
Anova(model_corr_meta)


# --- 4.2. Влияние корпусных метрик (LogDice, semantics) ---

# Замечаемость (visible)
model_vis_prop <- glmer(visible ~ LogDice_c + semantics_c + 
                          (1|participant) + (1|form) + (1|sentence),
                        data = human_joined, family = binomial,
                        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
summary(model_vis_prop)
Anova(model_vis_prop)

# Замечаемость ~ часть речи
model_vis_pos <- glmer(visible ~ pos + 
                         (1|participant) + (1|form) + (1|sentence),
                       data = human_joined, family = binomial,
                       control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
summary(model_vis_pos)
Anova(model_vis_pos)

# Замечаемость ~ паронимия
model_vis_par <- glmer(visible ~ paronyms + 
                         (1|participant) + (1|form) + (1|sentence),
                       data = human_joined, family = binomial,
                       control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
summary(model_vis_par)
Anova(model_vis_par)

# Приемлемость (acceptable) ~ LogDice + semantics
model_acc_prop <- glmer(acceptable ~ LogDice_c + semantics_c + 
                          (1|participant) + (1|form) + (1|sentence),
                        data = human_vis, family = binomial,
                        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
summary(model_acc_prop)
Anova(model_acc_prop)

# Точность (correctable) ~ LogDice + semantics
model_corr_prop <- glmer(correctable ~ LogDice_c + semantics_c + 
                           (1|participant) + (1|form) + (1|sentence),
                         data = human_acc, family = binomial,
                         control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
summary(model_corr_prop)
Anova(model_corr_prop)

# Точность (correctable) ~ часть речи
model_corr_pos <- glmer(correctable ~ pos + 
                          (1|participant) + (1|form) + (1|sentence),
                        data = human_acc, family = binomial,
                        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
summary(model_corr_pos)
Anova(model_corr_pos)

# Точность (correctable) ~ паронимия
model_corr_par <- glmer(correctable ~ paronyms + 
                          (1|participant) + (1|form) + (1|sentence),
                        data = human_acc, family = binomial,
                        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
summary(model_corr_par)
Anova(model_corr_par)


# --- 4.3. Проверка гипотезы об усталости (только Last-формы) ---

human_last <- human_joined[human_joined$last_form == TRUE, ]

model_vis_last <- glmer(visible ~ order_c + 
                          (1|participant) + (1|form) + (1|sentence),
                        data = human_last, family = binomial,
                        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
summary(model_vis_last)
Anova(model_vis_last)

# ============================================================================
# ЭТАП 5: КОРРЕЛЯЦИОННЫЙ АНАЛИЗ (ЛЮДИ)
# ============================================================================

human_agg <- human_joined %>%
  group_by(sentence) %>%
  summarise(
    visibility = mean(visible),
    precision_acceptable = mean(acceptable[visible == 1], na.rm = TRUE),
    recall_acceptable = mean(acceptable),
    precision_correctable = mean(correctable[acceptable == 1], na.rm = TRUE),
    recall_correctable = mean(correctable),
    LogDice = first(LogDice_right),
    semantics = first(semantics)
  )

# Корреляции с LogDice
cor.test(human_agg$LogDice, human_agg$visibility, method = "spearman")
cor.test(human_agg$LogDice, human_agg$precision_acceptable, method = "spearman")
cor.test(human_agg$LogDice, human_agg$recall_acceptable, method = "spearman")
cor.test(human_agg$LogDice, human_agg$precision_correctable, method = "spearman")
cor.test(human_agg$LogDice, human_agg$recall_correctable, method = "spearman")

# Корреляции с семантической близостью
cor.test(human_agg$semantics, human_agg$visibility, method = "spearman")
cor.test(human_agg$semantics, human_agg$precision_acceptable, method = "spearman")
cor.test(human_agg$semantics, human_agg$recall_acceptable, method = "spearman")
cor.test(human_agg$semantics, human_agg$precision_correctable, method = "spearman")
cor.test(human_agg$semantics, human_agg$recall_correctable, method = "spearman")

# ============================================================================
# ЭТАП 6: ВИЗУАЛИЗАЦИЯ ДАННЫХ ЛЮДЕЙ
# ============================================================================

# График: влияние образования на замечаемость
edu_plot <- human_joined %>%
  group_by(education) %>%
  summarise(prob = mean(visible), se = sqrt(prob*(1-prob)/n())) %>%
  filter(!is.na(education))

ggplot(edu_plot, aes(x = education, y = prob)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_errorbar(aes(ymin = prob - se, ymax = prob + se), width = 0.2) +
  labs(x = "Образование", y = "Visibility") +
  ylim(0, 1) + 
  theme_minimal()

# График: замечаемость vs LogDice
ggplot(human_agg, aes(x = LogDice, y = visibility)) +
  geom_point(size = 3, alpha = 0.7, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "darkred") +
  labs(x = "LogDice", y = "visibility") +
  theme_minimal()

# График: точность исправления vs семантическая близость
ggplot(human_agg, aes(x = semantics, y = precision_correctable)) +
  geom_point(size = 3, alpha = 0.7, color = "darkgreen") +
  geom_smooth(method = "lm", se = TRUE, color = "darkred") +
  labs(x = "Семантическая близость", y = "Исправляемость") +
  theme_minimal()

# График: эффект порядка предъявления (усталость)
last_agg <- human_last %>%
  group_by(order) %>%
  summarise(prob = mean(visible), se = sqrt(prob*(1-prob)/n()))

ggplot(last_agg, aes(x = order, y = prob)) +
  geom_point(size = 3, color = "steelblue") +
  geom_line(color = "steelblue", alpha = 0.5) +
  geom_errorbar(aes(ymin = prob - se, ymax = prob + se), width = 0.2) +
  geom_smooth(method = "lm", se = TRUE, color = "darkred") +
  labs(x = "Порядок предъявления", y = "Замечаемость") +
  ylim(0, 1) + 
  theme_minimal()

# ============================================================================
# ЭТАП 7: ПОДГОТОВКА И АНАЛИЗ ДАННЫХ МОДЕЛИ
# ============================================================================

# --- 7.1. Подготовка данных ---

model_joined <- R_model %>% left_join(R_sentences, by = "sentence")

model_joined$sentence <- as.factor(model_joined$sentence)
model_joined$temperature <- as.factor(model_joined$temperature)
model_joined$attempt <- as.factor(model_joined$attempt)
model_joined$pos <- as.factor(model_joined$pos)
model_joined$paronyms <- as.factor(model_joined$paronyms)

model_joined$LogDice_c <- scale(model_joined$LogDice_right, center = TRUE, scale = TRUE)
model_joined$semantics_c <- scale(model_joined$semantics, center = TRUE, scale = TRUE)
model_joined$additional <- ifelse(model_joined$add_inf != "" & !is.na(model_joined$add_inf), 1, 0)


# --- 7.2. Дескриптивная статистика модели ---

model_joined %>%
  group_by(temperature) %>%
  summarise(visible = mean(visible), acceptable = mean(acceptable), correctable = mean(correctable))

model_joined %>%
  group_by(attempt) %>%
  summarise(visible = mean(visible), acceptable = mean(acceptable), correctable = mean(correctable))


# --- 7.3. Регрессионный анализ модели ---

# Замечаемость (visible)
summary(glm(visible ~ LogDice_c + semantics_c, data = model_joined, family = binomial))
summary(glm(visible ~ pos, data = model_joined, family = binomial))
summary(glm(visible ~ paronyms, data = model_joined, family = binomial))

# Приемлемость (acceptable) — только на замеченных
model_acc_data <- model_joined[model_joined$visible == 1, ]
summary(glm(acceptable ~ LogDice_c + semantics_c, data = model_acc_data, family = binomial))

# Точность (correctable) — только на приемлемых
model_corr_data <- model_acc_data[model_acc_data$acceptable == 1, ]
summary(glm(correctable ~ LogDice_c + semantics_c, data = model_corr_data, family = binomial))
summary(glm(correctable ~ pos, data = model_corr_data, family = binomial))
summary(glm(correctable ~ paronyms, data = model_corr_data, family = binomial))


# --- 7.4. Влияние температуры и количества попыток ---

summary(glm(visible ~ temperature * attempt, data = model_joined, family = binomial))
summary(glm(visible ~ temperature, data = model_joined, family = binomial))
summary(glm(visible ~ attempt, data = model_joined, family = binomial))

# ============================================================================
# ЭТАП 8: ВИЗУАЛИЗАЦИЯ ДАННЫХ МОДЕЛИ
# ============================================================================

# Агрегированные данные
model_agg <- model_joined %>%
  group_by(sentence) %>%
  summarise(
    visibility = mean(visible),
    precision_correctable = mean(correctable[acceptable == 1], na.rm = TRUE),
    LogDice = first(LogDice_right),
    semantics = first(semantics),
    pos = first(pos)
  )

# График: заметность vs LogDice
p1 <- ggplot(model_agg, aes(x = LogDice, y = visibility)) +
  geom_point(size = 3, alpha = 0.7, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "darkred") +
  labs(x = "LogDice", y = "visibility") +
  ylim(0, 1) +
  theme_minimal()

# График: заметность vs семантическая близость
p2 <- ggplot(model_agg, aes(x = semantics, y = visibility)) +
  geom_point(size = 3, alpha = 0.7, color = "darkgreen") +
  geom_smooth(method = "lm", se = TRUE, color = "darkred") +
  labs(x = "Семантическая близость", y = "visibility") +
  ylim(0, 1) +
  theme_minimal()

# График: заметность vs часть речи
pos_vis <- model_joined %>%
  group_by(pos) %>%
  summarise(
    visibility = mean(visible, na.rm = TRUE),
    se = sqrt(visibility * (1 - visibility) / n())
  )

p3 <- ggplot(pos_vis, aes(x = pos, y = visibility, fill = pos)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_errorbar(aes(ymin = visibility - se, ymax = visibility + se), width = 0.2) +
  labs(x = "Часть речи", y = "visibility") +
  ylim(0, 1) +
  theme_minimal() +
  theme(legend.position = "none")

# Объединение графиков (заметность)
(p1 + p2) / p3 +
  plot_annotation(theme = theme(plot.title = element_text(hjust = 0.5)))


# --- 8.1. Графики точности попадания в золотой стандарт ---

# Агрегированные данные для correctable
model_agg_corr <- model_joined %>%
  filter(acceptable == 1) %>%
  group_by(sentence) %>%
  summarise(
    correctable_prop = mean(correctable, na.rm = TRUE),
    LogDice = first(LogDice_right),
    semantics = first(semantics),
    pos = first(pos)
  )

# График: correctable vs LogDice
p1_corr <- ggplot(model_agg_corr, aes(x = LogDice, y = correctable_prop)) +
  geom_point(size = 3, alpha = 0.7, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "darkred") +
  labs(x = "LogDice", y = "Исправляемость") +
  ylim(0, 1) +
  theme_minimal()

# График: correctable vs семантическая близость
p2_corr <- ggplot(model_agg_corr, aes(x = semantics, y = correctable_prop)) +
  geom_point(size = 3, alpha = 0.7, color = "darkgreen") +
  geom_smooth(method = "lm", se = TRUE, color = "darkred") +
  labs(x = "Семантическая близость", y = "Исправляемость") +
  ylim(0, 1) +
  theme_minimal()

# График: correctable vs часть речи
pos_corr <- model_joined %>%
  filter(acceptable == 1) %>%
  group_by(pos) %>%
  summarise(
    correctable_prop = mean(correctable, na.rm = TRUE),
    se = sqrt(correctable_prop * (1 - correctable_prop) / n())
  )

p3_corr <- ggplot(pos_corr, aes(x = pos, y = correctable_prop, fill = pos)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_errorbar(aes(ymin = correctable_prop - se, ymax = correctable_prop + se), width = 0.2) +
  labs(x = "Часть речи", y = "Исправляемость") +
  ylim(0, 1) +
  theme_minimal() +
  theme(legend.position = "none")

# Объединение графиков (исправляемость)
(p1_corr + p2_corr) / p3_corr +
  plot_annotation(theme = theme(plot.title = element_text(hjust = 0.5)))

# ============================================================================
# ЭТАП 9: СРАВНИТЕЛЬНЫЙ АНАЛИЗ ЛЮДЕЙ И МОДЕЛИ
# ============================================================================

# Подготовка данных для сравнения
human_common <- human_joined[, c("sentence", "visible", "acceptable", "correctable", 
                                 "LogDice_right", "semantics", "pos", "paronyms", "additional", "add_inf")]
human_common$source <- "human"

model_common <- model_joined[, c("sentence", "visible", "acceptable", "correctable", 
                                 "LogDice_right", "semantics", "pos", "paronyms", "additional", "add_inf")]
model_common$source <- "model"

all_data <- bind_rows(human_common, model_common)
all_data$source <- as.factor(all_data$source)
all_data$LogDice_c <- scale(all_data$LogDice_right, center = TRUE, scale = TRUE)
all_data$semantics_c <- scale(all_data$semantics, center = TRUE, scale = TRUE)

# Регрессионное сравнение: замечаемость (visible)
summary(glm(visible ~ source * LogDice_c, data = all_data, family = binomial))
summary(glm(visible ~ source * semantics_c, data = all_data, family = binomial))

# Регрессионное сравнение: точность (correctable) — только на приемлемых
all_data_acc <- all_data[all_data$acceptable == 1, ]
summary(glm(correctable ~ source * LogDice_c, data = all_data_acc, family = binomial))
summary(glm(correctable ~ source * semantics_c, data = all_data_acc, family = binomial))


# --- 9.1. График сравнения метрик ---

metrics_comp <- data.frame(
  source = rep(c("human", "model"), each = 3),
  metric = rep(c("visibility", "precision_acceptability", "precision_correctability"), 2),
  value = c(
    mean(all_data$visible[all_data$source == "human"]),
    mean(all_data$acceptable[all_data$source == "human" & all_data$visible == 1], na.rm = TRUE),
    mean(all_data$correctable[all_data$source == "human" & all_data$acceptable == 1], na.rm = TRUE),
    mean(all_data$visible[all_data$source == "model"]),
    mean(all_data$acceptable[all_data$source == "model" & all_data$visible == 1], na.rm = TRUE),
    mean(all_data$correctable[all_data$source == "model" & all_data$acceptable == 1], na.rm = TRUE)
  )
)

metrics_comp$metric <- factor(metrics_comp$metric, 
                              levels = c("visibility", "precision_acceptability", "precision_correctability"))

ggplot(metrics_comp, aes(x = metric, y = value, fill = source)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
  labs(x = "Метрика", y = "Доля", fill = "Источник") +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format()) +
  scale_x_discrete(labels = c("visibility" = "Замечаемость\n(visible)",
                              "precision_acceptability" = "Приемлемость\n(precision_acc)",
                              "precision_correctability" = "Точность\n(precision_corr)")) +
  theme_minimal() +
  theme(legend.position = "bottom")

# ============================================================================
# ЭТАП 10: АНАЛИЗ ГИПЕРКОРРЕКЦИИ (additional)
# ============================================================================

# Расчёт долей
human_additional <- human_joined %>%
  summarise(
    n_total = n(),
    n_additional = sum(additional, na.rm = TRUE),
    prop_additional = n_additional / n_total
  )

model_additional <- model_joined %>%
  summarise(
    n_total = n(),
    n_additional = sum(additional, na.rm = TRUE),
    prop_additional = n_additional / n_total
  )

additional_comparison <- bind_rows(
  human_additional %>% mutate(source = "human"),
  model_additional %>% mutate(source = "model")
)

print("=== Склонность к гиперкоррекции (additional == 1) ===")
print(additional_comparison)

# Статистическая значимость различий
additional_test <- bind_rows(
  human_joined %>% select(additional) %>% mutate(source = "human"),
  model_joined %>% select(additional) %>% mutate(source = "model")
)

add_table <- table(additional_test$source, additional_test$additional)
print("Таблица сопряжённости (source x additional):")
print(add_table)

chisq_test <- chisq.test(add_table)
print(paste("Chi-square test: p =", round(chisq_test$p.value, 5)))

# Типы дополнительных правок
cat("\n=== ТИПЫ ДОПОЛНИТЕЛЬНЫХ ИСПРАВЛЕНИЙ У ЛЮДЕЙ ===\n")
human_joined %>%
  filter(additional == 1) %>%
  group_by(add_inf) %>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>%
  print(n = 10)

cat("\n=== ТИПЫ ДОПОЛНИТЕЛЬНЫХ ИСПРАВЛЕНИЙ У МОДЕЛИ ===\n")
model_joined %>%
  filter(additional == 1) %>%
  group_by(add_inf) %>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>%
  print(n = 10)

# ============================================================================
# ЭТАП 11: АНАЛИЗ ЭНТРОПИИ (ВАРИАТИВНОСТЬ ИСПРАВЛЕНИЙ)
# ============================================================================

# --- 11.1. Функция расчёта энтропии ---

calc_entropy <- function(x) {
  x <- x[!is.na(x) & x != ""]
  if(length(x) == 0) return(NA)
  probs <- table(x) / length(x)
  -sum(probs * log(probs))
}


# --- 11.2. Энтропия для людей ---

human_acc_ent <- human_joined[human_joined$acceptable == 1, ]
human_acc_ent$response_cat <- ifelse(human_acc_ent$correctable == 1, "GOLD", human_acc_ent$acceptable_inf)
human_acc_ent <- human_acc_ent[human_acc_ent$response_cat != "" & !is.na(human_acc_ent$response_cat), ]

entropy_human <- data.frame()
for (sent in unique(human_acc_ent$sentence)) {
  sub <- human_acc_ent[human_acc_ent$sentence == sent, "response_cat"]
  if(length(sub) == 0) next
  
  dist <- table(sub)
  entropy_human <- rbind(entropy_human, data.frame(
    sentence = sent,
    n_raters = length(sub),
    n_categories = length(dist),
    entropy = calc_entropy(sub),
    gold_prop = ifelse("GOLD" %in% names(dist), dist["GOLD"] / sum(dist), 0)
  ))
}

entropy_human$sentence <- as.integer(entropy_human$sentence)
R_sentences$sentence <- as.integer(R_sentences$sentence)

entropy_human <- entropy_human %>%
  left_join(R_sentences, by = "sentence")

# Корреляции энтропии (люди)
cor.test(entropy_human$entropy, entropy_human$semantics, method = "spearman")
cor.test(entropy_human$entropy, entropy_human$LogDice_right, method = "spearman")


# --- 11.3. Энтропия для модели ---

model_acc_ent <- model_joined[model_joined$acceptable == 1, ]
model_acc_ent$response_cat <- ifelse(model_acc_ent$correctable == 1, "GOLD", model_acc_ent$acceptable_inf)
model_acc_ent <- model_acc_ent[model_acc_ent$response_cat != "" & !is.na(model_acc_ent$response_cat), ]

entropy_model <- data.frame()
for (sent in unique(model_acc_ent$sentence)) {
  sub <- model_acc_ent[model_acc_ent$sentence == sent, "response_cat"]
  if(length(sub) == 0) next
  
  dist <- table(sub)
  entropy_model <- rbind(entropy_model, data.frame(
    sentence = sent,
    entropy = calc_entropy(sub),
    gold_prop = ifelse("GOLD" %in% names(dist), dist["GOLD"] / sum(dist), 0)
  ))
}

cat("\n=== СРАВНЕНИЕ ЭНТРОПИИ ===\n")
cat("Средняя энтропия у людей:", mean(entropy_human$entropy, na.rm = TRUE), "\n")
cat("Средняя энтропия у модели:", mean(entropy_model$entropy, na.rm = TRUE), "\n")


# --- 11.4. Предложения, где модель никогда не попадает в золотой стандарт ---

model_no_gold <- entropy_model[entropy_model$gold_prop == 0, ]
human_sporadic <- entropy_human[entropy_human$entropy > 1.0, ]

intersection <- merge(model_no_gold, human_sporadic, by = "sentence")
print("Предложения, где модель не попадает в gold, а у людей высокая энтропия:")
print(intersection)


# --- 11.5. Вывод энтропии по каждому предложению ---

cat("\n=== ЭНТРОПИЯ ПО ПРЕДЛОЖЕНИЯМ (ВСЕ) ===\n")
entropy_human %>%
  select(sentence, entropy, n_categories, gold_prop, LogDice_right, semantics, paronyms) %>%
  arrange(desc(entropy)) %>%
  print()

cat("\n=== ТОП-10 ПРЕДЛОЖЕНИЙ С НАИБОЛЬШЕЙ ЭНТРОПИЕЙ ===\n")
entropy_human %>%
  arrange(desc(entropy)) %>%
  select(sentence, entropy, n_categories, gold_prop, LogDice_right, semantics, paronyms) %>%
  head(10) %>%
  print()

cat("\n=== ТОП-10 ПРЕДЛОЖЕНИЙ С НАИМЕНЬШЕЙ ДОЛЕЙ GOLD ===\n")
entropy_human %>%
  arrange(gold_prop) %>%
  select(sentence, gold_prop, entropy, n_categories, LogDice_right, semantics, paronyms) %>%
  head(10) %>%
  print()


# --- 11.6. Графики энтропии ---

# График: энтропия vs семантическая близость
p1 <- ggplot(entropy_human, aes(x = semantics, y = entropy)) +
  geom_point(size = 3, alpha = 0.7, color = "darkgreen") +
  geom_smooth(method = "lm", se = TRUE, color = "darkred", fill = "pink") +
  labs(subtitle = paste0("Spearman's ρ = -0.492, p = 0.004"),
       x = "Семантическая близость", y = "Энтропия (H)") +
  theme_minimal()

# График: энтропия vs LogDice
p2 <- ggplot(entropy_human, aes(x = LogDice_right, y = entropy)) +
  geom_point(size = 3, alpha = 0.7, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "darkred", fill = "pink") +
  labs(subtitle = paste0("Spearman's ρ = -0.335, p = 0.061"),
       x = "LogDice_right", y = "Энтропия (H)") +
  theme_minimal()

# Объединение графиков
(p1 + p2) +
  plot_annotation(theme = theme(plot.title = element_text(hjust = 0.5)))

# График с подписями номеров предложений
ggplot(entropy_human, aes(x = semantics, y = entropy, label = sentence)) +
  geom_point(size = 3, alpha = 0.7, color = "darkgreen") +
  geom_text(vjust = -0.8, size = 3) +
  geom_smooth(method = "lm", se = TRUE, color = "darkred", fill = "pink") +
  labs(title = "Энтропия vs семантическая близость",
       subtitle = paste0("Spearman's ρ = -0.492, p = 0.004"),
       x = "Семантическая близость", y = "Энтропия (H)") +
  theme_minimal()

# Линейная регрессия: что предсказывает энтропию?
model_entropy <- lm(entropy ~ LogDice_right + semantics + paronyms, 
                    data = entropy_human)
summary(model_entropy)
