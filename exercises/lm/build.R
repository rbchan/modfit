library(rmarkdown)

# render("lm.Rmd", output_format=html_document(self_contained=FALSE))

render("lm.Rmd", output_format=html_document(self_contained=FALSE),
       output_dir="../../_includes")
