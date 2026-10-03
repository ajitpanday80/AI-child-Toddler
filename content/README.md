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

## Story collections with morals (Panchatantra, Jataka, fables ...)
Write (or keep) each story followed by its lesson. The school finds the lessons by themselves, in `.txt`, `.md`, PDF or web-page text:

```
The Thirsty Crow

Once a crow was very thirsty. He found a jug with a little water at the bottom ... He dropped pebbles into the jug until the water rose.

Moral: Where there is a will, there is a way.
```
Accepted lead-ins (any capitalisation): `Moral: ...`, `Lesson - ...`, `The moral of the story is ...`. The lesson runs to the next blank line.
A short title line above the story is optional. Stories shorter than about 150 characters, or lessons shorter than 8 characters, are read as ordinary text.

What happens to them:
- They are studied as **story -> moral** lessons (the model learns to write `Moral: ...` after a story), filed under *values*.
- About 20% are held back for the **moral exam** (pick the true moral of an unseen story out of 4). The `moral=` score appears in the exam line **only for the stage whose folder contains the stories**, so put them in the folder of the stage you are training now (for example `content/grade2/`). It is shown for information; it does not gate promotion.
- In the chat, `/moral <story>` asks the model for a moral.
- More stories help: a few hundred is a start; the more the better.
