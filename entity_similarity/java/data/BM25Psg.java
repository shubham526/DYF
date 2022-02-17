package data;

import help.RankingHelper;
import org.apache.lucene.document.Document;
import org.jetbrains.annotations.NotNull;
import java.util.List;

public class BM25Psg extends CandidatePsg{

    public BM25Psg(String paraIndex,
                        String entityIndex,
                        String entityParaFile,
                        String stopWordsFile,
                        String dataFile,
                        String entityFile,
                        boolean parallel) {

        super(paraIndex, entityIndex, entityParaFile, stopWordsFile, dataFile, entityFile, parallel);
    }

    @Override
    protected String getEntityDescription(String entityId, @NotNull List<RankingHelper.ScoredDocument> rankedParaList) {
        Document topDoc = rankedParaList.get(0).getDocument();
        String topDocText = String.join(
                " ",
                RankingHelper.preProcess(
                        topDoc.get("Text"),
                        stopWords
                ));
        return topDocText;
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

        BM25Psg ob = new BM25Psg(
                paraIndex, entityIndex, entityParaFile, stopWordsFile, dataFile, entityFile, parallel
        );

        ob.doTask(outFile);
    }

}
