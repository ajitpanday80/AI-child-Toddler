# Your own teaching material

Drop material for a stage into its folder. It is added to that stage's lessons
(and held-out parts of it are used in that stage's exam).

```
content/
  values/  pre-nursery/  nursery/  grade1/  grade2/ ... grade6/
      my_book.pdf            <- PDF: text is extracted
      notes.txt  notes.md    <- plain text
      links.txt              <- a list of web pages or PDF links, one per line (lines starting with # are ignored)
      math/worksheet.pdf     <- optional subfolder = subject: values language math science history civics mind
                                (anything else is filed under "extra")
```
- A `.txt` file where **every** line is a `http(s)://` link is treated as a list of links; the pages (or PDFs behind them) are downloaded and read.
- Scanned PDFs (pictures of pages) contain no text and are skipped. Use text PDFs.
- Everything is cached in `data/`, so links are fetched once.
- Only add material you are allowed to use.
