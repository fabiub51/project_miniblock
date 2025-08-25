##### Statistical Analyses #####
library(tidyverse)
library(ggpubr)
library(rstatix)
library(lme4)
library(lmerTest)
library(broom.mixed)
library(readr)

##### Reliability #####
reliability_results_all <- read_csv("reliability_results_all.csv")
for (d in unique(reliability_results_all$ROI)){
  print(d)
  smoothed_rel <- reliability_results_all %>% 
    filter(smoothing == "sm_2_vox", ROI == d) %>% 
    group_by(subject, ROI, runtype) %>% 
    summarize(median_reliability = mean(median_reliability)) %>% 
    ungroup()
  res.aov <- aov(median_reliability ~ runtype + Error(subject/runtype), data = smoothed_rel)
  print(summary(res.aov))
}

for (d in unique(reliability_results_all$ROI)){
  print(d)
  smoothed_rel <- reliability_results_all %>% 
    filter(smoothing == "unsmoothed", ROI == d) %>% 
    group_by(subject, ROI, runtype) %>% 
    summarize(median_reliability = mean(median_reliability)) %>% 
    ungroup()
  
  res.aov <- aov(median_reliability ~ runtype + Error(subject/runtype), data = smoothed_rel)
  print(summary(res.aov))
}

p_results_smoothed <- list()
for (d in unique(reliability_results_all$ROI)){
  smoothed_rel <- reliability_results_all %>%
    filter(smoothing == "sm_2_vox", ROI == d) %>%
    group_by(subject, ROI, runtype) %>%
    summarize(median_reliability = mean(median_reliability, na.rm = TRUE), .groups = "drop")%>%
    pivot_wider(names_from = runtype, values_from = median_reliability)
  p_value_er_mini <- t.test(smoothed_rel$er, smoothed_rel$miniblock, paired = TRUE)$p.value
  p_fdr_er_mini <- p.adjust(p_value_er_mini, method = "fdr")
  p_value_er_sus <- t.test(smoothed_rel$er, smoothed_rel$sus, paired = TRUE)$p.value
  p_fdr_er_sus <- p.adjust(p_value_er_sus, method = "fdr")
  p_value_sus_mini <- t.test(smoothed_rel$sus, smoothed_rel$miniblock, paired = TRUE)$p.value
  p_fdr_sus_mini <- p.adjust(p_value_sus_mini, method = "fdr")
  
  p_results_smoothed[[d]] <- list(
    er_vs_mini = p_fdr_er_mini,
    er_vs_sus = p_fdr_er_sus,
    sus_vs_mini = p_fdr_sus_mini
  )
}

p_results_unsmoothed <- list()
for (d in unique(reliability_results_all$ROI)){
  smoothed_rel <- reliability_results_all %>%
    filter(smoothing == "unsmoothed", ROI == d) %>%
    group_by(subject, ROI, runtype) %>%
    summarize(median_reliability = mean(median_reliability, na.rm = TRUE), .groups = "drop")%>%
    pivot_wider(names_from = runtype, values_from = median_reliability)
  p_value_er_mini <- t.test(smoothed_rel$er, smoothed_rel$miniblock, paired = TRUE)$p.value
  p_fdr_er_mini <- p.adjust(p_value_er_mini, method = "fdr")
  p_value_er_sus <- t.test(smoothed_rel$er, smoothed_rel$sus, paired = TRUE)$p.value
  p_fdr_er_sus <- p.adjust(p_value_er_sus, method = "fdr")
  p_value_sus_mini <- t.test(smoothed_rel$sus, smoothed_rel$miniblock, paired = TRUE)$p.value
  p_fdr_sus_mini <- p.adjust(p_value_sus_mini, method = "fdr")
  
  p_results_unsmoothed[[d]] <- list(
    er_vs_mini = p_fdr_er_mini,
    er_vs_sus = p_fdr_er_sus,
    sus_vs_mini = p_fdr_sus_mini
  )
}

##### Noise Ceilings #####
df_noise_ceilings <- read_csv("df_noise_ceilings.csv", col_types = cols(...1 = col_skip()))

for (d in unique(df_noise_ceilings$ROI)){
  print(d)
  smoothed_noise <- df_noise_ceilings %>% 
    filter(smoothing == "sm_2_vox", ROI == d) %>% 
    group_by(subject, ROI, runtype) %>% 
    summarize(median_nc = mean(median_nc)) %>% 
    ungroup()
  res.aov <- aov(median_nc ~ runtype + Error(subject/runtype), data = smoothed_noise)
  print(summary(res.aov))
}

for (d in unique(df_noise_ceilings$ROI)){
  print(d)
  unsmoothed_noise <- df_noise_ceilings %>% 
    filter(smoothing == "unsmoothed", ROI == d) %>% 
    group_by(subject, ROI, runtype) %>% 
    summarize(median_nc = mean(median_nc)) %>% 
    ungroup()
  res.aov <- aov(median_nc ~ runtype + Error(subject/runtype), data = unsmoothed_noise)
  print(summary(res.aov))
}

p_results_smoothed <- list()
for (d in unique(df_noise_ceilings$ROI)){
  smoothed_rel <- df_noise_ceilings %>%
    filter(smoothing == "sm_2_vox", ROI == d) %>%
    group_by(subject, ROI, runtype) %>%
    summarize(median_nc = mean(median_nc, na.rm = TRUE), .groups = "drop")%>%
    pivot_wider(names_from = runtype, values_from = median_nc)
  p_value_er_mini <- t.test(smoothed_rel$er, smoothed_rel$miniblock, paired = TRUE)$p.value
  p_fdr_er_mini <- p.adjust(p_value_er_mini, method = "fdr")
  p_value_er_sus <- t.test(smoothed_rel$er, smoothed_rel$sus, paired = TRUE)$p.value
  p_fdr_er_sus <- p.adjust(p_value_er_sus, method = "fdr")
  p_value_sus_mini <- t.test(smoothed_rel$sus, smoothed_rel$miniblock, paired = TRUE)$p.value
  p_fdr_sus_mini <- p.adjust(p_value_sus_mini, method = "fdr")
  
  p_results_smoothed[[d]] <- list(
    er_vs_mini = p_fdr_er_mini,
    er_vs_sus = p_fdr_er_sus,
    sus_vs_mini = p_fdr_sus_mini
  )
}

p_results_unsmoothed <- list()
for (d in unique(df_noise_ceilings$ROI)){
  smoothed_rel <- df_noise_ceilings %>%
    filter(smoothing == "unsmoothed", ROI == d) %>%
    group_by(subject, ROI, runtype) %>%
    summarize(median_nc = mean(median_nc, na.rm = TRUE), .groups = "drop")%>%
    pivot_wider(names_from = runtype, values_from = median_nc)
  p_value_er_mini <- t.test(smoothed_rel$er, smoothed_rel$miniblock, paired = TRUE)$p.value
  p_fdr_er_mini <- p.adjust(p_value_er_mini, method = "fdr")
  p_value_er_sus <- t.test(smoothed_rel$er, smoothed_rel$sus, paired = TRUE)$p.value
  p_fdr_er_sus <- p.adjust(p_value_er_sus, method = "fdr")
  p_value_sus_mini <- t.test(smoothed_rel$sus, smoothed_rel$miniblock, paired = TRUE)$p.value
  p_fdr_sus_mini <- p.adjust(p_value_sus_mini, method = "fdr")
  
  p_results_unsmoothed[[d]] <- list(
    er_vs_mini = p_fdr_er_mini,
    er_vs_sus = p_fdr_er_sus,
    sus_vs_mini = p_fdr_sus_mini
  )
}

##### Decoding #####
decoding_ROI_data <- read_csv("decoding_ROI_data.csv", col_types = cols(...1 = col_skip()))
decoding_ROI_group_data <- read_csv("decoding_ROI_group_data.csv")

# Check for above chance decoding 
for (r in unique(decoding_ROI_data$ROI)){
  for (d in unique(decoding_ROI_data$Design)){
    print(r)
    print(d)
    decoding_filtered <- decoding_ROI_data %>% 
      filter(ROI == r, 
             Design == d) %>% 
      group_by(Subject, ROI, Design) %>% 
      summarize(mean_decoding = mean(Accuracy)) %>% 
      ungroup()
    significant_decoding <- t.test(decoding_filtered$mean_decoding, mu = 50)
    print(significant_decoding$p.value*5)}}

# RM-ANOVA
for (d in unique(decoding_ROI_data$ROI)){
  print(d)
  decoding_filtered <- decoding_ROI_data %>% 
    filter(ROI == d) %>% 
    group_by(Subject, ROI, Design) %>% 
    summarize(mean_decoding = mean(Accuracy)) %>% 
    ungroup()
  res.aov <- aov(mean_decoding ~ Design + Error(Subject/Design), data = decoding_filtered)
  print(summary(res.aov))
}

# Post-hoc t-tests
p_decoding <- list()
for (d in unique(decoding_ROI_data$ROI)){
  decoding_filtered <- decoding_ROI_data %>%
    filter(ROI == d) %>%
    group_by(Subject, ROI, Design) %>%
    summarize(mean_decoding = mean(Accuracy, na.rm = TRUE),.groups = "drop")%>%
    pivot_wider(names_from = Design, values_from = mean_decoding)
  p_value_er_mini <- t.test(decoding_filtered$er, decoding_filtered$miniblock, paired = TRUE)$p.value
  p_fdr_er_mini <- p.adjust(p_value_er_mini, method = "fdr")
  p_value_er_sus <- t.test(decoding_filtered$er, decoding_filtered$sus, paired = TRUE)$p.value
  p_fdr_er_sus <- p.adjust(p_value_er_sus, method = "fdr")
  p_value_sus_mini <- t.test(decoding_filtered$sus, decoding_filtered$miniblock, paired = TRUE)$p.value
  p_fdr_sus_mini <- p.adjust(p_value_sus_mini, method = "fdr")
  
  p_decoding[[d]] <- list(
    er_vs_mini = p_fdr_er_mini,
    er_vs_sus = p_fdr_er_sus,
    sus_vs_mini = p_fdr_sus_mini
  )
}

# Animate objects
for (d in unique(decoding_ROI_data_group$ROI)){
  print(d)
  decoding_filtered <- decoding_ROI_data_group %>% 
    filter(group == "isanimate") %>% 
    filter(ROI == d) %>% 
    group_by(Subject, ROI, Design) %>% 
    summarize(
      mean_decoding = mean(Accuracy), 
      sd_decoding = sd(Accuracy),
      .groups = "drop"
    ) %>% 
    ungroup()
  res.aov <- aov(mean_decoding ~ Design + Error(Subject/Design), data = decoding_filtered)
  print(summary(res.aov))
}

# Post-hoc t-tests
p_decoding <- list()
for (d in unique(decoding_ROI_data_group$ROI)){
  decoding_filtered <- decoding_ROI_data_group %>% 
    filter(group == "isanimate") %>% 
    filter(ROI == d) %>% 
    group_by(Subject, ROI, Design) %>% 
    summarize(
      mean_decoding = mean(Accuracy), 
      sd_decoding = sd(Accuracy),
      .groups = "drop"
    )%>%
    pivot_wider(names_from = Design, values_from = mean_decoding)
  p_value_er_mini <- t.test(decoding_filtered$er, decoding_filtered$miniblock, paired = TRUE)$p.value
  p_fdr_er_mini <- p.adjust(p_value_er_mini, method = "fdr")
  p_value_er_sus <- t.test(decoding_filtered$er, decoding_filtered$sus, paired = TRUE)$p.value
  p_fdr_er_sus <- p.adjust(p_value_er_sus, method = "fdr")
  p_value_sus_mini <- t.test(decoding_filtered$sus, decoding_filtered$miniblock, paired = TRUE)$p.value
  p_fdr_sus_mini <- p.adjust(p_value_sus_mini, method = "fdr")
  
  p_decoding[[d]] <- list(
    er_vs_mini = p_fdr_er_mini,
    er_vs_sus = p_fdr_er_sus,
    sus_vs_mini = p_fdr_sus_mini
  )
}

# Inanimate objects
for (d in unique(decoding_ROI_data_group$ROI)){
  print(d)
  decoding_filtered <- decoding_ROI_data_group %>% 
    filter(group == "inanimate") %>% 
    filter(ROI == d) %>% 
    group_by(Subject, ROI, Design) %>% 
    summarize(
      mean_decoding = mean(Accuracy), 
      sd_decoding = sd(Accuracy),
      .groups = "drop"
    ) %>% 
    ungroup()
  res.aov <- aov(mean_decoding ~ Design + Error(Subject/Design), data = decoding_filtered)
  print(summary(res.aov))
}

# Post-hoc t-tests
p_decoding <- list()
for (d in unique(decoding_ROI_data_group$ROI)){
  decoding_filtered <- decoding_ROI_data_group %>% 
    filter(group == "inanimate") %>% 
    filter(ROI == d) %>% 
    group_by(Subject, ROI, Design) %>% 
    summarize(
      mean_decoding = mean(Accuracy), 
      sd_decoding = sd(Accuracy),
      .groups = "drop"
    )%>%
    pivot_wider(names_from = Design, values_from = mean_decoding)
  p_value_er_mini <- t.test(decoding_filtered$er, decoding_filtered$miniblock, paired = TRUE)$p.value
  p_fdr_er_mini <- p.adjust(p_value_er_mini, method = "fdr")
  p_value_er_sus <- t.test(decoding_filtered$er, decoding_filtered$sus, paired = TRUE)$p.value
  p_fdr_er_sus <- p.adjust(p_value_er_sus, method = "fdr")
  p_value_sus_mini <- t.test(decoding_filtered$sus, decoding_filtered$miniblock, paired = TRUE)$p.value
  p_fdr_sus_mini <- p.adjust(p_value_sus_mini, method = "fdr")
  
  p_decoding[[d]] <- list(
    er_vs_mini = p_fdr_er_mini,
    er_vs_sus = p_fdr_er_sus,
    sus_vs_mini = p_fdr_sus_mini
  )
}

# Scenes
for (d in unique(decoding_ROI_data_group$ROI)){
  print(d)
  decoding_filtered <- decoding_ROI_data_group %>% 
    filter(group == "scene") %>% 
    filter(ROI == d) %>% 
    group_by(Subject, ROI, Design) %>% 
    summarize(
      mean_decoding = mean(Accuracy), 
      sd_decoding = sd(Accuracy),
      .groups = "drop"
    ) %>% 
    ungroup()
  res.aov <- aov(mean_decoding ~ Design + Error(Subject/Design), data = decoding_filtered)
  print(summary(res.aov))
}

# Post-hoc t-tests
p_decoding <- list()
for (d in unique(decoding_ROI_data_group$ROI)){
  decoding_filtered <- decoding_ROI_data_group %>% 
    filter(group == "scene") %>% 
    filter(ROI == d) %>% 
    group_by(Subject, ROI, Design) %>% 
    summarize(
      mean_decoding = mean(Accuracy), 
      sd_decoding = sd(Accuracy),
      .groups = "drop"
    )%>%
    pivot_wider(names_from = Design, values_from = mean_decoding)
  p_value_er_mini <- t.test(decoding_filtered$er, decoding_filtered$miniblock, paired = TRUE)$p.value
  p_fdr_er_mini <- p.adjust(p_value_er_mini, method = "fdr")
  p_value_er_sus <- t.test(decoding_filtered$er, decoding_filtered$sus, paired = TRUE)$p.value
  p_fdr_er_sus <- p.adjust(p_value_er_sus, method = "fdr")
  p_value_sus_mini <- t.test(decoding_filtered$sus, decoding_filtered$miniblock, paired = TRUE)$p.value
  p_fdr_sus_mini <- p.adjust(p_value_sus_mini, method = "fdr")
  
  p_decoding[[d]] <- list(
    er_vs_mini = p_fdr_er_mini,
    er_vs_sus = p_fdr_er_sus,
    sus_vs_mini = p_fdr_sus_mini
  )
}

# RSA - between participants
rsa_results_spearman_between <- read_csv("~/Documents/project_miniblock/miniblock/Outputs/RSA/ROI_between/rsa_results_spearman_between.csv")

# RM-ANOVA
for (d in unique(rsa_results_spearman_between$ROI)){
  print(d)
  rsa_between_filtered <- rsa_results_spearman_between %>% 
    filter(ROI == d) %>% 
    group_by(pair, ROI, runtype) %>% 
    summarize(mean_correlation= mean(correlation)) %>% 
    ungroup()
  res.aov <- aov(mean_correlation ~ runtype + Error(pair/runtype), data = rsa_between_filtered)
  print(summary(res.aov))
}

# Post-hoc t-tests
p_rsa_between <- list()
for (d in unique(rsa_results_spearman_between$ROI)){
  rsa_between_filtered <- rsa_results_spearman_between %>%
    filter(ROI == d) %>%
    group_by(pair, ROI, runtype) %>%
    summarize(mean_correlation = mean(correlation, na.rm = TRUE),.groups = "drop")%>%
    pivot_wider(names_from = runtype, values_from = mean_correlation)
  p_value_er_mini <- t.test(rsa_between_filtered$er, rsa_between_filtered$miniblock, paired = TRUE)$p.value
  p_fdr_er_mini <- p.adjust(p_value_er_mini, method = "fdr")
  p_value_er_sus <- t.test(rsa_between_filtered$er, rsa_between_filtered$sus, paired = TRUE)$p.value
  p_fdr_er_sus <- p.adjust(p_value_er_sus, method = "fdr")
  p_value_sus_mini <- t.test(rsa_between_filtered$sus, rsa_between_filtered$miniblock, paired = TRUE)$p.value
  p_fdr_sus_mini <- p.adjust(p_value_sus_mini, method = "fdr")
  
  p_rsa_between[[d]] <- list(
    er_vs_mini = p_fdr_er_mini,
    er_vs_sus = p_fdr_er_sus,
    sus_vs_mini = p_fdr_sus_mini
  )
}

##### EVC analysis #####
evc_results <- read_csv("evc_results.csv")

for (d in unique(evc_results$ROI)){
  print(d)
  smoothed_evc <- evc_results %>% 
    filter(ROI == d) %>% 
    group_by(subject, ROI, design) %>% 
    summarize(mean_correlation = mean(correlation)) %>% 
    ungroup()
  res.aov <- aov(mean_correlation ~ design + Error(subject/design), data = smoothed_evc)
  print(summary(res.aov))
}

##### Reliability Within Runs #####
reliability_progression_within <- read_csv("reliability_progression_within.csv")

results <- list()

for (s in c("sm_2_vox", "unsmoothed")) {
  for (d in unique(reliability_progression_within$runtype)) {
    for (r in unique(reliability_progression_within$ROI)) {
      
      model <- lmer(
        median_rel ~ run + (1|subject),
        data = subset(reliability_progression_within, smoothing == s & runtype == d & ROI == r)
      )
      
      tidy_mod <- broom.mixed::tidy(model) %>% 
        filter(term == "run") %>% 
        mutate(runtype = d, ROI = r, smoothing = s)
      
      results[[length(results)+1]] <- tidy_mod
    }
  }
}

results_df <- bind_rows(results)

# Apply FDR correction separately per design
results_df <- results_df %>%
  group_by(runtype, smoothing) %>%
  mutate(p_adj = p.adjust(p.value, method = "fdr")) %>%
  ungroup()

results_df %>% 
  filter(p_adj < 0.05)

wide_R2_data <- R2_df %>% 
  select(ROI, runtype, component, value) %>% 
  pivot_wider(names_from = c(ROI,runtype), values_from = value) %>% 
  mutate(across(c(starts_with("PPA"), starts_with("FFA"), starts_with("EBA"), starts_with("EVC")), 
                ~ c(.x[1], diff(.x)),
                .names = "{.col}_ind"))%>% 
  select(component,ends_with("_ind"))

##### Cross-Validated PCA #####
R2_df <- read_csv("R2_df.csv")
wide_R2_data %>%
  pivot_longer(
    cols = -component,                    # keep 'component' as is
    names_to = c("ROI", "runtype"),       # split into ROI and runtype
    names_sep = "_",                       # split at underscore
    values_to = "value"                    # column for values
  ) %>% 
  ggplot(aes(x=component, y=value, color = as.factor(runtype))) +
  geom_line(linewidth=1.1)+
  facet_wrap(~as.factor(ROI))+
  labs(
    title = "Cross-Validated Principle Components",
    x = "Number of Components",
    y = "Mean Explained Variance in Test Set",
    color = "Runtype"
  ) +
  scale_y_log10()+
  scale_x_log10()+
  scale_color_brewer(palette = "Set2") +
  my_theme()

ggsave("PCA_CV_slopes.jpg", last_plot(), height = 10, width = 15)


