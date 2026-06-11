# ReCOGS Info

- ReCOGS is a semantic parsing dataset used for testing compositional generalization on a fragment of English.
- We are using the main split from `ReCOGS/main/recogs_positional_index`
- Raw files are stored in: `experiments/recogs/data/raw/`
- Format: TSV, no header, 3 columns (`input`, `logical_form`, `category`)

## size/ split
- `train.tsv`: **27,227** rows
- `dev.tsv`: **3,000** rows
- `test.tsv`: **3,000** rows
- `gen.tsv`: **21,000** rows

## Sample Examples

### train (in_distribution)
- input: `A rose was helped by a dog .`
- lf: `rose ( 1 ) ; dog ( 6 ) ; help ( 3 ) AND theme ( 3 , 1 ) AND agent ( 3 , 6 )`
- tag: `in_distribution`

### dev (in_distribution)
- input: `Liam hoped that a box was burned by a girl .`
- lf: `Liam ( 0 ) ; box ( 4 ) ; girl ( 9 ) ; hope ( 1 ) AND agent ...`
- tag: `in_distribution`

### test (in_distribution)
- input: `The moose wanted to read .`
- lf: `* moose ( 1 ) ; want ( 2 ) AND agent ( 2 , 1 ) AND xcomp ( 2 , 4 ) AND read ( 4 ) AND agent ( 4 , 1 )`
- tag: `in_distribution`

### gen (OOD lexical)
- input: `James rolled Paula .`
- lf: `James ( 0 ) ; Paula ( 2 ) ; roll ( 1 ) AND agent ( 1 , 0 ) AND theme ( 1 , 2 )`
- tag: `prim_to_obj_proper`

### gen (OOD role shift)
- input: `A hero painted the hedgehog beside a house .`
- lf: `hero ( 1 ) ; * hedgehog ( 4 ) ; house ( 7 ) ; paint ( 2 ) AND agent ...`
- tag: `subj_to_obj_common`

### gen (OOD structural recursion)
- input: `The queen froze the strawberry in the cabinet ... beside the book .`
- lf: `* queen ( 1 ) ; * strawberry ( 4 ) ; ... AND nmod . in ... AND nmod . beside ...`
- tag: `pp_recursion`