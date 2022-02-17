package data;

import help.LuceneHelper;
import help.RankingHelper;
import help.Utilities;
import org.apache.lucene.document.Document;
import org.apache.lucene.search.BooleanQuery;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.json.JSONException;
import org.json.JSONObject;
import java.util.ArrayList;
import java.util.List;

public abstract class CandidatePsg extends CreateEmbeddingData{

    public CandidatePsg(String paraIndex,
                   String entityIndex,
                   String entityParaFile,
                   String stopWordsFile,
                   String dataFile,
                   String entityFile,
                   boolean parallel) {

        super(paraIndex, entityIndex, entityParaFile, stopWordsFile, dataFile, entityFile, parallel);
    }

    @Nullable
    @Override
    protected String getEntityDescription(String sentenceContext, String entityId, String entityName) {
        try {
            // Get the paragraphs which mention the entity
            if (entityParaMap.containsKey(entityId)) {
                List<String> paraList = Utilities.JSONArrayToList(new JSONObject(entityParaMap.get(entityId))
                        .getJSONArray("paragraphs"));

                // Rank these paragraphs for the query
                List<RankingHelper.ScoredDocument> rankedParaList = rankParasForQuery(sentenceContext, entityName, paraList);

                if (!rankedParaList.isEmpty()) {
                    return getEntityDescription(entityId, rankedParaList);
                }else {
                    System.err.println("No ranked paragraphs found for entity: " + entityId);
                }
            }
            else {
                System.out.println("No paragraphs found for entity: " + entityId);
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }

        return null;
    }

    @NotNull
    protected List<RankingHelper.ScoredDocument> rankParasForQuery(String sentenceContext,
                                                                 String entityName,
                                                                 List<String> paraList) {

        // Get the Lucene documents
        List<Document> luceneDocList = LuceneHelper.toLuceneDocList(paraList, paraIndexSearcher);

        // Convert to BooleanQuery
        BooleanQuery booleanQuery = RankingHelper.toBooleanQueryWithPRF(
                sentenceContext,
                entityName,
                luceneDocList,
                stopWords
        );

        // Rank the Lucene Documents using the BooleanQuery

        if (booleanQuery == null) {
            return new ArrayList<>();
        }

        return RankingHelper.rankDocuments(booleanQuery, luceneDocList, 1000);
    }

    protected abstract String getEntityDescription(String entityId, List<RankingHelper.ScoredDocument> rankedParaList);

}
