library(rmarkdown)


render("lm.Rmd", "html_document")
##render("lm.Rmd", "md_document")
render("lm.Rmd", md_document(variant = "markdown"))
