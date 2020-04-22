library("testthat")
library("Andromeda")

context("test-isSorted.R")

test_that("isSorted data.frame", {
  x <- data.frame(a = runif(1000),b = runif(1000))
  x <- round(x,digits=2)
  expect_false(isSorted(x,c("a","b"),c(TRUE,FALSE)))
  x <- x[order(x$a,x$b),]

  expect_true(isSorted(x,c("a","b")))
  expect_false(isSorted(x,c("a","b"),c(TRUE,FALSE)))

  x <- x[order(x$a,-x$b),]
  expect_true(isSorted(x,c("a","b"),c(TRUE,FALSE)))
  expect_false(isSorted(x,c("a","b")))
})

test_that("isSorted Andromeda", {
#   x <- data.frame(a = runif(20000000),b = runif(20000000)) # Takes too much time for a unit-test
  x <- data.frame(a = runif(200),b = runif(200))
  x <- round(x,digits=2)
  andr <- andromeda(x = x)
  expect_false(isSorted(andr$x,c("a","b"),c(TRUE,FALSE)))

  x <- x[ffdforder(x[c("a","b")]),]

  expect_true(isSorted(x,c("a","b")))
  expect_false(isSorted(x,c("a","b"),c(TRUE,FALSE)))

  x$minb <- 0-x$b
  x <- x[ffdforder(x[c("a","minb")]),]
  expect_true(isSorted(x,c("a","b"),c(TRUE,FALSE)))
  expect_false(isSorted(x,c("a","b")))
})
