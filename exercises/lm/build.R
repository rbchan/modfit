library(rmarkdown)

## render("lm.Rmd", output_format=html_document(self_contained=TRUE),
##        output_dir="../../_includes")

## render("lm.Rmd", output_format=html_document(self_contained=FALSE),
##        output_dir="../../_includes")



render("lm.Rmd",
       output_format=html_fragment(mathjax=TRUE),
       ##html_fragment(self_contained=TRUE),
       output_dir="../../_includes")
