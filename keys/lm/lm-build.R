## Create .tex file
library(knitr)
knit("lm-key.Rnw")

## Compile and open PDF
tools::texi2pdf("lm-key.tex")
system("open lm-key.pdf")


## It seems we can't easily convert our .Rnw or .tex to .html
## but here's a try with pandoc
## Convert .tex file to .html and .md
system("pandoc -s lm-key.tex --mathjax -o lm-key.html")


## A tedious workaround is to create a .Rmd and then manually insert
## the R code where it's missing
## We could just start with .Rmd, but markdown isn't as flexible as Latex
## Uncomment the next region if you want to do this
## system("pandoc -s lm-key.tex --mathjax -t markdown -o lm-key.Rmd")

## Once the .Rmd file is ready, you can build html like this
library(rmarkdown)
render(input="lm-key.Rmd", output_format="github_document")
render("lm-key.Rmd", "html_document")









## BAD IDEAS BELOW


## Convert the .R file to .html
knit("lm-key.Rnw", tangle=TRUE)
##render("lm-key.R", "all")
render("lm-key.R", "md_document")

##render("lm-key.Rnw", "html_document")


## Try with htlatex (fails?)
##system("htlatex lm-key.tex 'xhtml, mathml, charset=utf-8' ' -cunihtf -utf8'")
