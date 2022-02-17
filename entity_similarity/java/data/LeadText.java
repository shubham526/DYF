package data;

import help.LuceneHelper;
import org.apache.lucene.document.Document;
import org.apache.lucene.queryparser.classic.ParseException;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import java.io.IOException;


public class LeadText extends CreateEmbeddingData{
    public LeadText (String paraIndex,
                       String entityIndex,
                       String entityParaFile,
                       String stopWordsFile,
                       String dataFile,
                       String entityFile,
                       boolean parallel) {
        super(paraIndex, entityIndex, entityParaFile, stopWordsFile, dataFile, entityFile, parallel);
    }

    @Override
    @Nullable
    protected String getEntityDescription(String sentenceContext, String entityId, String entityName) {
        try {
            Document doc = LuceneHelper.searchIndex("Id", entityId, entityIndexSearcher);
            if (doc != null) {
                return doc.get("LeadText");
            }
        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void main(@NotNull String[] args) {
        String paraIndex = args[0];
        String entityIndex = args[1];
        String entityParaFile = args[2];
        String stopWordsFile = args[3];
        String dataFile = args[4];
        String entityFile = args[5];
        String outFile = args[6];
        boolean parallel = args[7].equals("true");

        LeadText ob = new LeadText(
                paraIndex, entityIndex, entityParaFile, stopWordsFile, dataFile, entityFile, parallel
        );

        ob.doTask(outFile);
    }
}
