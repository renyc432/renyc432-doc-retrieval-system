To use these indexes:
1. Download pyserini from github
2. Put the index folder of your choice at '\pyserini-master\indexes'


Create new index from documents:
1. Check out this guide: https://github.com/castorini/pyserini#how-do-i-search-my-own-documents
2. Download pyserini from github
3. Go to '\pyserini-master\pyserini\resources\jars' and put the fatjar files there
3. Go to '\pyserini-master\integrations\resources' and create a folder with your json files inside
3. Run
python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator 
-threads 1 -input integrations/resources/index_name 
-index indexes/index_name -storePositions -storeDocvectors -storeRaw
