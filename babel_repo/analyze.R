# Revenue analysis helpers.

monthly_revenue <- function(orders) {
  aggregate(orders$total, by = list(orders$month), FUN = sum)
}

plot_revenue = function(rev) {
  barplot(rev$x)
}
