# R-side companion to utils.py — shared functions for the R and Quarto stages (00-04).
#
# Source it from a script or a Quarto chunk:
#     source(here::here("src", "R_utils.R"))

library(dplyr)

#' Attach a year-indexed panel to observations, falling back to the latest available year.
#'
#' The project's one rule for matching a year (doc/decisions.md D32): use the source's value for
#' the exact year when it has one, otherwise the most recent *earlier* year for that key; if the
#' source begins after the requested year, use its earliest. This mirrors
#' `utils.join_latest_available_year()` exactly, so the population behind a 2025 survey and the
#' population behind a 2025 regional weight are chosen the same way.
#'
#' The year actually used is returned in `used_year_col`, so a carried-forward value is never
#' silent — compare it against `year` to find them.
#'
#' @param target observations to attach to; must carry `key` and `year`
#' @param source year-indexed panel; must carry `key` and `year`, one row per pair
#' @param key column identifying the unit, typically "iso3"
#' @param year the year column, present in both
#' @param used_year_col name for the source year actually used
#' @return `target` with the source's remaining columns and `used_year_col` attached; rows whose
#'   key is absent from `source` keep NA, exactly as a left join would leave them
join_latest_available_year <- function(target, source, key = "iso3", year = "year",
                                       used_year_col = paste0(year, "_used")) {
  dup <- source %>% count(.data[[key]], .data[[year]]) %>% filter(n > 1)
  if (nrow(dup)) {
    stop(sprintf("source has %d duplicated %s-%s rows", nrow(dup), key, year), call. = FALSE)
  }

  src <- source %>% rename(!!used_year_col := all_of(year))

  # Candidate source years per target row, then keep the closest one at or before the target year;
  # where none exists (the target predates the source), keep the earliest instead.
  picks <- target %>%
    distinct(.data[[key]], .data[[year]]) %>%
    inner_join(src %>% select(all_of(c(key, used_year_col))), by = key, relationship = "many-to-many") %>%
    group_by(.data[[key]], .data[[year]]) %>%
    slice_min(
      order_by = ifelse(.data[[used_year_col]] <= .data[[year]],
                        .data[[year]] - .data[[used_year_col]],           # earlier: prefer nearest
                        Inf),
      n = 1, with_ties = FALSE
    ) %>%
    # every candidate was later than the target year -> fall back to the source's earliest
    mutate(!!used_year_col := .data[[used_year_col]]) %>%
    ungroup()

  # slice_min on an all-Inf group keeps an arbitrary row, so redo those explicitly.
  too_late <- picks %>% filter(.data[[used_year_col]] > .data[[year]]) %>% select(all_of(c(key, year)))
  if (nrow(too_late)) {
    earliest <- src %>% group_by(.data[[key]]) %>%
      summarise(!!used_year_col := min(.data[[used_year_col]]), .groups = "drop")
    picks <- picks %>%
      anti_join(too_late, by = c(key, year)) %>%
      bind_rows(too_late %>% left_join(earliest, by = key))
  }

  target %>%
    left_join(picks, by = c(key, year)) %>%
    left_join(src, by = c(key, used_year_col))
}


#' Which rows did not get their own year — the audit that keeps the fallback visible.
carried_year_report <- function(frame, key = "iso3", year = "year",
                                used_year_col = paste0(year, "_used")) {
  frame %>%
    filter(!is.na(.data[[used_year_col]]), .data[[used_year_col]] != .data[[year]]) %>%
    transmute(!!key := .data[[key]], !!year := .data[[year]],
              !!used_year_col := .data[[used_year_col]],
              lag = .data[[year]] - .data[[used_year_col]]) %>%
    arrange(.data[[year]], .data[[key]])
}
