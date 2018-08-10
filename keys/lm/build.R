library(rmarkdown)

## render("lm.Rmd", output_format=html_document(self_contained=TRUE),
##        output_dir="../../_includes")

## render("lm.Rmd", output_format=html_document(self_contained=FALSE),
##        output_dir="../../_includes")



out.file <- render("lm-key.Rmd",
                   output_format=html_fragment(
                       mathjax=TRUE,
                       ## includes =
                       ##     includes(before_body="mathjax.html"),
                       self_contained=TRUE),
                   output_dir="../../_includes")

lm.html.in <- readLines(out.file)
mathjax.in <- readLines("../../mathjax.html")
lm.html.out <- c(mathjax.in, lm.html.in)
writeLines(lm.html.out, out.file)
