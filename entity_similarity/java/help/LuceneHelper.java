package help;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.LMJelinekMercerSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LuceneHelper {

    @Nullable
    public static Similarity getSimilarity(@NotNull String similarityStr) {

        if (similarityStr.equalsIgnoreCase("bm25")) {
            return new BM25Similarity();
        } else if (similarityStr.equalsIgnoreCase("lmds")) {
            return new LMDirichletSimilarity(1500);
        } else if (similarityStr.equalsIgnoreCase("lmjm")) {
            return new LMJelinekMercerSimilarity(0.5f);
        }
        return null;

    }

    @NotNull
    public static IndexSearcher createSearcher(String indexDir, @NotNull String similarityStr) {
        Similarity similarity = getSimilarity(similarityStr);

        Directory dir = null;
        try {
            dir = FSDirectory.open((new File(indexDir).toPath()));
        } catch (IOException e) {
            e.printStackTrace();
        }
        IndexReader reader = null;
        try {
            reader = DirectoryReader.open(dir);
        } catch (IOException e) {
            e.printStackTrace();
        }
        assert reader != null;
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(similarity);
        return searcher;
    }

    @Nullable
    public static Document searchIndex(String field, String query, @NotNull IndexSearcher searcher)throws IOException, ParseException {
        Term term = new Term(field,query);
        Query q = new TermQuery(term);
        TopDocs tds = searcher.search(q,1);

        ScoreDoc[] retDocs = tds.scoreDocs;
        if(retDocs.length != 0) {
            return searcher.doc(retDocs[0].doc);
        }
        return null;
    }

    @NotNull
    public static List<Document> toLuceneDocList(@NotNull List<String> paraList, IndexSearcher indexSearcher) {
        List<Document> documentList = new ArrayList<>();
        for (String paraId : paraList) {
            try {
                Document d = searchIndex("Id", paraId, indexSearcher);
                if (d != null) {
                    documentList.add(d);
                }
            } catch (IOException | ParseException e) {
                e.printStackTrace();
            }

        }
        return documentList;
    }

    /**
     * This class is based on the Lucene BytesBuffersDirectory.
     * @version 11/26/2021
     */

    public static class MemoryIndex {

        @NotNull
        @Contract(value = " -> new", pure = true)
        public static Directory initialize() {
            return new ByteBuffersDirectory();
        }

        public static IndexWriter createWriter(Directory dir, Analyzer analyzer) {
            IndexWriterConfig conf = new IndexWriterConfig(analyzer);
            conf.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
            IndexWriter iw = null;
            try {
                iw = new IndexWriter(dir, conf);
            } catch (IOException e) {
                e.printStackTrace();
            }
            return iw;

        }

        @NotNull
        public static IndexSearcher createSearcher(Directory dir, Similarity similarity) throws IOException {
            IndexReader reader = DirectoryReader.open(dir);
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(similarity);
            return searcher;
        }
        /**
         * Build an in-memory index of documents passed as parameters.
         * @param documents The documents to index
         */
        public static void createIndex(@NotNull List<Document> documents, IndexWriter iw) throws IOException {
            for (Document d : documents) {
                if (d != null) {
                    try {
                        iw.addDocument(d);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            iw.commit();
        }
        public static void close(@NotNull IndexWriter iw) throws IOException {
            iw.getDirectory().close();

        }
    }



}
