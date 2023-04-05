# Gets run when compiling

makevars_in <- file.path("src", "Makevars.in")
makevars_win_in <- file.path("src", "Makevars.win.in")

makevars_out <- file.path("src", "Makevars")
makevars_win_out <- file.path("src", "Makevars.win")

txt <- readLines(makevars_in)
txt_win <- readLines(makevars_win_in)

if (getRversion() < "4.1") {
    if (!any(grepl("^CXX_STD", txt))) {
        txt <- c("CXX_STD = CXX11", txt)
    }

    if (!any(grepl("^CXX_STD", txt_win))) {
        txt_win <- c("CXX_STD = CXX11", txt_win)
    }
}

cat(txt, file = makevars_out, sep = "\n")
cat(txt_win, file = makevars_win_out, sep = "\n")
