package data;

import help.RankingHelper;
import help.Utilities;
import org.apache.lucene.document.Document;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SupportPsg extends CandidatePsg {

    public SupportPsg (String paraIndex,
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
    protected String getEntityDescription(String entityId, @NotNull List<RankingHelper.ScoredDocument> rankedParaList) {

        // Create the ECD using the ranked paragraphs
        Utilities.EntityContextDocument d = createECD(entityId, rankedParaList);
        if (d != null) {
            List<String> contextEntityList = d.getEntityList();
            Map<String, Integer> freqDist = getDistribution(contextEntityList);
            freqDist.remove(entityId);
            return getSupportPsgForEntity(d, freqDist);
        }
        return null;
    }


    @NotNull
    protected String getSupportPsgForEntity(@NotNull Utilities.EntityContextDocument d, Map<String, Integer> freqMap) {

        // Get the list of documents in the pseudo-document corresponding to the entity
        List<Document> documents = d.getDocumentList();
        Map<String, Integer> scoreMap = scoreSupportPsg(documents, freqMap);
        Map.Entry<String, Integer> topSupportPsgForEntity = new ArrayList<>(scoreMap.entrySet()).get(0);
        String topSupportPsgId = topSupportPsgForEntity.getKey();
        return Utilities.idToText(topSupportPsgId, "Text", paraIndexSearcher);

    }

    @NotNull
    private Map<String, Integer> scoreSupportPsg(@NotNull List<Document> documents, Map<String, Integer> freqMap) {
        Map<String, Integer> scoreMap = new HashMap<>();
        // For every document do
        for (Document doc : documents) {

            // Get the paragraph id of the document
            String paraId = doc.getField("Id").stringValue();

            // Get the score of the document
            int score = getParaScore(doc, freqMap);

            // Store the paragraph id and score in a HashMap
            scoreMap.put(paraId, score);
        }
        return Utilities.sortByValueDescending(scoreMap);
    }

    protected int getParaScore(@NotNull Document doc, Map<String, Integer> freqMap) {

        int entityScore, paraScore = 0;
        // Get the entities in the paragraph
        // Make an ArrayList from the String array

        List<String> entityList = Utilities.getEntitiesInPara(doc);
        /* For every entity in the paragraph do */
        for (String e : entityList) {
            // Lookup this entity in the HashMap of frequencies for the entities
            // Sum over the scores of the entities to get the score for the passage
            // Store the passage score in the HashMap
            if (freqMap.containsKey(e)) {
                entityScore = freqMap.get(e);
                paraScore += entityScore;
            }

        }
        return paraScore;
    }

    @NotNull
    private Map<String, Integer> getDistribution(@NotNull List<String> contextEntityList) {
        HashMap<String, Integer> freqMap = new HashMap<>();

        // For every co-occurring entity do
        for (String entityID : contextEntityList) {
            freqMap.compute(entityID, (t, oldV) -> (oldV == null) ? 1 : oldV + 1);
        }
        return  Utilities.sortByValueDescending(freqMap);
    }

    @Nullable
    protected Utilities.EntityContextDocument createECD(String entityId,
                                                        @NotNull List<RankingHelper.ScoredDocument> paraList) {
        List<Document> documentList = new ArrayList<>();
        List<String> contextEntityList = new ArrayList<>();
        for (RankingHelper.ScoredDocument scoredDocument : paraList) {
            Document doc = scoredDocument.getDocument();
            List<String> entityList = Utilities.getEntitiesInPara(doc);
            if (entityList.isEmpty()) {
                // If the document does not have any entities then ignore
                continue;
            }
            if (entityList.contains(entityId)) {
                documentList.add(doc);
                contextEntityList.addAll(entityList);
            } else{
                System.out.println("Target entity not in document.");
            }
        }

        // If there are no documents in the pseudo-document
        if (documentList.size() == 0) {
            return null;
        }
        return new Utilities.EntityContextDocument(documentList, entityId, contextEntityList);
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

        SupportPsg ob = new SupportPsg(
                paraIndex, entityIndex, entityParaFile, stopWordsFile, dataFile, entityFile, parallel
        );

        ob.doTask(outFile);
    }
}
