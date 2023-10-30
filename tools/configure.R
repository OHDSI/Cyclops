# Gets run when compiling

makevars_in <- file.path("src", "Makevars.in")
makevars_win_in <- file.path("src", "Makevars.win.in")

makevars_out <- file.path("src", "Makevars")
makevars_win_out <- file.path("src", "Makevars.win")

txt <- readLines(makevars_in)
txt_win <- readLines(makevars_win_in)

if (getRversion() < "4.3") { # macOS / linux
    if (!any(grepl("^CXX_STD", txt))) {
        txt <- c("CXX_STD = CXX11", txt)
    }
}

if (getRversion() < "4.2") { # Windoz
    if (!any(grepl("^CXX_STD", txt_win))) {
        txt_win <- c("CXX_STD = CXX11", txt_win)
    }
}

if (.Platform$OS.type == "unix") {
	cat(txt, file = makevars_out, sep = "\n")
} else {
	cat(txt_win, file = makevars_win_out, sep = "\n")
}
