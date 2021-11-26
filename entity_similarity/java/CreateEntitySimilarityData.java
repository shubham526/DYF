import help.LuceneHelper;
import help.RankingHelper;
import help.Utilities;
import me.tongfei.progressbar.ProgressBar;
import org.apache.lucene.document.Document;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import java.util.Random;

import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;


public class CreateEntitySimilarityData {
    private final Map<String, String> entityParaMap; // Map containing (entity_id, List(para_id)) where para_id --> contains link to entity_id
    private final Map<String, Set<String>> entityRankings;
    private final Map<String, String> queryIdToNameMap;
    private  final Map<String, String> entityIdToNameMap;
    private final List<String> stopWords;
    String dataType;
    private final List<String> dataStrings = new ArrayList<>();
    private final IndexSearcher indexSearcher;
    private int total = 0;
    private final int topK;
    private final Map<String, Set<String>> entities;
    private final AtomicInteger count = new AtomicInteger(0);
    private final boolean parallel;


    /**
     * Class to represent an Entity Context Document for an entity.
     * @author Shubham Chatterjee
     * @version 05/31/2020
     */
    public static class EntityContextDocument {

        private final List<Document> documentList;
        private final String entity;
        private final List<String> contextEntities;

        /**
         * Constructor.
         * @param documentList List of documents in the pseudo-document
         * @param entity The entity for which the pseudo-document is made
         * @param contextEntities The list of entities in the pseudo-document
         */
        @Contract(pure = true)
        public EntityContextDocument(List<Document> documentList,
                                     String entity,
                                     List<String> contextEntities) {
            this.documentList = documentList;
            this.entity = entity;
            this.contextEntities = contextEntities;
        }

        /**
         * Method to get the list of documents in the ECD.
         * @return String
         */
        public List<Document> getDocumentList() {
            return this.documentList;
        }

        /**
         * Method to get the entity of the ECD.
         * @return String
         */
        public String getEntity() {
            return this.entity;
        }

        /**
         * Method to get the list of context entities in the ECD.
         * @return ArrayList
         */
        public List<String> getEntityList() {
            return this.contextEntities;
        }
    }

    public static final class Pair<K, V> implements Map.Entry<K, V> {
        private final K key;
        private V value;

        public Pair(K key, V value) {
            this.key = key;
            this.value = value;
        }

        @Override
        public K getKey() {
            return key;
        }

        @Override
        public V getValue() {
            return value;
        }

        @Override
        public V setValue(V value) {
            V old = this.value;
            this.value = value;
            return old;
        }
    }



    public CreateEntitySimilarityData(String paraIndex,
                                      String entityParaFile,
                                      String entityRunFile,
                                      String entityFile,
                                      String queryIdToNameFile,
                                      String entityIdToNameFile,
                                      String stopWordsFile,
                                      int topK,
                                      String dataType,
                                      String outFile,
                                      boolean parallel) {

        this.parallel = parallel;
        this.topK = topK;
        this.dataType = dataType;

        System.out.print("Setting up paragraph index...");
        this.indexSearcher = LuceneHelper.createSearcher(paraIndex, "bm25");
        System.out.println("[Done].");

        System.out.print("Loading entity file...");
        entities = Utilities.readEntityFile(entityFile);
        System.out.println("[Done].");

        System.out.print("Loading entity to passage mappings...");
        entityParaMap = Utilities.readTsvFile(entityParaFile);
        System.out.println("[Done].");

        System.out.print("Loading entity rankings...");
        entityRankings = Utilities.readEntityFile(entityRunFile);
        System.out.println("[Done].");

        System.out.print("Loading QueryId to QueryName mappings...");
        queryIdToNameMap = Utilities.readTsvFile(queryIdToNameFile);
        System.out.println("[Done].");

        System.out.print("Loading EntityId to EntityName mappings...");
        entityIdToNameMap = Utilities.readTsvFile(entityIdToNameFile);
        System.out.println("[Done].");

        System.out.print("Loading stop words....");
        stopWords = Utilities.getStopWords(stopWordsFile);
        System.out.println("[Done].");

        doTask();

        System.out.print("Writing to file....");
        Utilities.writeToFile(outFile, dataStrings);
        System.out.println("[Done].");

    }

    public void doTask() {
        Set<String> querySet = entityRankings.keySet();
        total = querySet.size();

        if (parallel) {
            System.out.println("Using Parallel Streams.");
            int parallelism = ForkJoinPool.commonPool().getParallelism();
            int numOfCores = Runtime.getRuntime().availableProcessors();
            System.out.println("Number of available processors = " + numOfCores);
            System.out.println("Number of threads generated = " + parallelism);

            if (parallelism == numOfCores - 1) {
                System.err.println("WARNING: USING ALL AVAILABLE PROCESSORS");
                System.err.println("USE: \"-Djava.util.concurrent.ForkJoinPool.common.parallelism=N\" " +
                        "to set the number of threads used");
            }
            // Do in parallel
            querySet.parallelStream().forEach(this::getEntityData);
        } else {
            System.out.println("Using Sequential Streams.");

            // Do in serial
            ProgressBar pb = new ProgressBar("Progress", querySet.size());
            for (String q : querySet) {
                getEntityData(q);
                pb.step();
            }
            pb.close();
        }
    }

    public void getEntityData(@NotNull String queryId) {

        if (entityRankings.containsKey(queryId)) {
            Set<String> candidateEntitySet = entities.get(queryId);

            for (String entityId : candidateEntitySet) {
                Pair<String, Map<String, Integer>> entityWithSupportPsg = getSupportPsgForEntity(queryId, entityId,  true);
                String supportPsg = entityWithSupportPsg.getKey();
                Map<String, String> relatedEntities = getRelatedEntities(queryId, entityWithSupportPsg.getValue());
                Map<String, String> nonRelatedEntities = getNonRelatedEntities(queryId);
                if (dataType.equals("pointwise")) {
                    toPointWiseData(supportPsg, relatedEntities, nonRelatedEntities);
                } else {
                    toPairWiseData(supportPsg, relatedEntities, nonRelatedEntities);
                }
            }
            if (parallel) {
                count.getAndIncrement();
                System.out.println("Done: " + queryId + " ( " + count + "/" + total + " ).");
            }
        }
    }

    private void toPairWiseData(String supportPsg,
                                @NotNull Map<String, String> relatedEntities,
                                @NotNull Map<String, String> nonRelatedEntities) {
        List<String> relList = new ArrayList<>(relatedEntities.values());
        List<String> nonRelList = new ArrayList<>(nonRelatedEntities.values());
        List<Pair<String, String>> pairs = relList.stream()
                .flatMap(i -> nonRelList.stream()
                        .map(j -> new Pair<>(i, j)))
                .collect(Collectors.toList());

        for (Pair<String, String> pair : pairs) {
            dataStrings.add(toJSONString(String.join(" ", RankingHelper.preProcess(supportPsg, stopWords)),
                    String.join(" ", RankingHelper.preProcess(pair.getKey(), stopWords)),
                    String.join(" ", RankingHelper.preProcess(pair.getValue(), stopWords))
            ));
        }

    }

    private void toPointWiseData(String supportPsg,
                                 @NotNull Map<String, String> relatedEntities,
                                 Map<String, String> nonRelatedEntities) {
        for (String entityId : relatedEntities.keySet()) {
            dataStrings.add(toJSONString(String.join(" ", RankingHelper.preProcess(supportPsg, stopWords)),
                    String.join(" ", RankingHelper.preProcess(relatedEntities.get(entityId), stopWords)),
                    1
            ));
        }
        for (String entityId : nonRelatedEntities.keySet()) {
           if (nonRelatedEntities.get(entityId) != null) {
               dataStrings.add(toJSONString(String.join(" ", RankingHelper.preProcess(supportPsg, stopWords)),
                       String.join(" ", RankingHelper.preProcess(nonRelatedEntities.get(entityId), stopWords)),
                       0
               ));
           }

        }
    }


    @NotNull
    private Pair<String, Map<String, Integer>> getSupportPsgForEntity(String queryId,
                                                        @NotNull String entityId,
                                                        boolean returnFreqDist) {


        Set<String> retEntitySet = entityRankings.get(queryId);
        try {
            // Get the paragraphs which mention the entity
            List<String> paraList = JSONArrayToList(new JSONObject(entityParaMap.get(entityId))
                    .getJSONArray("paragraphs"));

            // Rank these paragraphs for the query
            List<RankingHelper.ScoredDocument> rankedParaList = rankParasForQuery(queryId, entityId, paraList);

            if (!rankedParaList.isEmpty()) {

                // Create the ECD using the ranked paragraphs
                EntityContextDocument d = createECD(entityId, rankedParaList);
                if (d != null) {
                    List<String> contextEntityList = d.getEntityList();
                    Map<String, Integer> freqDist = getDistribution(contextEntityList, retEntitySet);
                    freqDist.remove(entityId);
                    String supportPsgText = getSupportPsgForEntity(d, freqDist);
                    if  (returnFreqDist) {
                        return new Pair<>(supportPsgText, freqDist);
                    } else {
                        return new Pair<>(supportPsgText, null);
                    }
                }
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }

        return new Pair<>(null, null);

    }

    @NotNull
    private Map<String, String> getRelatedEntities(String queryId,
                                                   @NotNull Map<String, Integer> freqDist) {

        Map<String, String> relEntityMap = new HashMap<>();
        List<Map.Entry<String, Integer>> allEntities = new ArrayList<>(freqDist.entrySet());
        List<Map.Entry<String, Integer>> topKEntities = allEntities.subList(0,
                Math.min(topK, allEntities.size()));

        for (Map.Entry<String, Integer> entry : topKEntities) {
            String entityId = entry.getKey();
            Pair<String, Map<String, Integer>> entityWithSupportPsg = getSupportPsgForEntity(queryId, entityId, false);
            relEntityMap.put(entityId, entityWithSupportPsg.getKey());
        }
        return relEntityMap;
    }

    @NotNull
    private Map<String, String> getNonRelatedEntities(String queryId) {
        Map<String, String> nonRelEntityMap = new HashMap<>();
        List<Map.Entry<String, Set<String>>> allEntities = new ArrayList<>(entityRankings.entrySet());
        int randomIndex;

        // We will generate an index at random and use the entities for that query as non-related entities
        do {
            randomIndex = new Random().nextInt(allEntities.size());
        } while (allEntities.get(randomIndex).getKey().equals(queryId));

        Map.Entry<String, Set<String>> randomEntry = allEntities.get(randomIndex);
        String randomQuery = randomEntry.getKey();
        Set<String> nonRelEntitySet = allEntities.get(randomIndex).getValue();
        List<String> topKEntities = new ArrayList<>(nonRelEntitySet).subList(0, Math.min(topK, nonRelEntitySet.size()));

        for (String entityId : topKEntities) {
            Pair<String, Map<String, Integer>> entityWithSupportPsg = getSupportPsgForEntity(randomQuery, entityId, false);
            nonRelEntityMap.put(entityId, entityWithSupportPsg.getKey());
        }

        return nonRelEntityMap;
    }

    @NotNull
    protected List<RankingHelper.ScoredDocument> rankParasForQuery(String queryId, String entityId, List<String> paraList) {

        // Get the Lucene documents
        List<Document> luceneDocList = LuceneHelper.toLuceneDocList(paraList, indexSearcher);

        // Convert to BooleanQuery
        BooleanQuery booleanQuery = RankingHelper.toBooleanQueryWithPRF(queryIdToNameMap.get(queryId),
                entityIdToNameMap.get(entityId), luceneDocList, stopWords);

        // Rank the Lucene Documents using the BooleanQuery

        if (booleanQuery == null) {
            return new ArrayList<>();
        }

        return RankingHelper.rankDocuments(booleanQuery, luceneDocList, 1000);

    }


    @NotNull
    protected List<String> JSONArrayToList(@NotNull JSONArray paragraphs) {
        List<String> result = new ArrayList<>();
        for (int i = 0; i < paragraphs.length(); i++) {
            try {
                result.add(paragraphs.getString(i));
            } catch (JSONException e) {
                e.printStackTrace();
            }
        }
        return result;
    }

    @Nullable
    protected EntityContextDocument createECD(String entityId,
                                              @NotNull List<RankingHelper.ScoredDocument> paraList) {
        List<Document> documentList = new ArrayList<>();
        List<String> contextEntityList = new ArrayList<>();
        for (RankingHelper.ScoredDocument scoredDocument : paraList) {
            Document doc = scoredDocument.getDocument();
            List<String> entityList = getEntitiesInPara(doc);
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
        return new EntityContextDocument(documentList, entityId, contextEntityList);
    }

    @NotNull
    protected List<String> getEntitiesInPara(@NotNull Document doc) {
        List<String> entityList = new ArrayList<>();
        String[] paraEntities = doc.get("Entities").split("\n");

        for (String entity : paraEntities) {
            if (! entity.isEmpty()) {
                try {
                    JSONObject jsonObject = new JSONObject(entity);
                    String linkedEntityId = jsonObject.getString("linkPageId");
                    entityList.add(linkedEntityId);
                } catch (JSONException e) {
                    e.printStackTrace();
                }

            }
        }
        return entityList;
    }

    @NotNull
    protected Map<String, Integer> getDistribution(@NotNull List<String> contextEntityList,
                                                  Set<String> retEntitySet) {

        HashMap<String, Integer> freqMap = new HashMap<>();

        // For every co-occurring entity do
        for (String entityID : contextEntityList) {
            // If the entity also occurs in the list of entities retrieved for the query then
            if ( retEntitySet.contains(entityID)) {
                freqMap.compute(entityID, (t, oldV) -> (oldV == null) ? 1 : oldV + 1);
            }
        }
        return  Utilities.sortByValueDescending(freqMap);
    }



    @NotNull
    protected String getSupportPsgForEntity(@NotNull EntityContextDocument d, Map<String, Integer> freqMap) {

        // Get the list of documents in the pseudo-document corresponding to the entity
        List<Document> documents = d.getDocumentList();
        Map<String, Integer> scoreMap = scoreSupportPsg(documents, freqMap);
        Map.Entry<String, Integer> topSupportPsgForEntity = new ArrayList<>(scoreMap.entrySet()).get(0);
        String topSupportPsgId = topSupportPsgForEntity.getKey();
        String topSupportPsgText = Utilities.idToText(topSupportPsgId, "Text", indexSearcher);
        return topSupportPsgText;

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

    /**
     * Method to find the score of a paragraph.
     * This method looks at all the entities in the paragraph and calculates the score from them.
     * For every entity in the paragraph, if the entity has a score from the entity context pseudo-document,
     * then sum over the entity scores and store the score in a HashMap.
     *
     * @param doc  Document
     * @param freqMap HashMap where Key = entity id and Value = score
     * @return Integer
     */

    protected int getParaScore(@NotNull Document doc, Map<String, Integer> freqMap) {

        int entityScore, paraScore = 0;
        // Get the entities in the paragraph
        // Make an ArrayList from the String array

        List<String> entityList = getEntitiesInPara(doc);
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

    public static String toJSONString(String query, String docPos, String docNeg) {
        JSONObject example = new JSONObject();
        try {
            example.put("query", query);
            example.put("doc_pos", docPos);
            example.put("doc_neg", docNeg);
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return example.toString();
    }

    public static String toJSONString(String query, String doc, int label) {
        JSONObject example = new JSONObject();
        try {
            example.put("query", query);
            example.put("doc", doc);
            example.put("label", label);
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return example.toString();
    }



    public static void main(@NotNull String[] args) {

        String paraIndex = args[0];
        String entityPassageFile = args[1];
        String entityRunFile = args[2];
        String entityFile = args[3];
        String queryIdToNameFile = args[4];
        String entityIdToNameFile = args[5];
        String stopWordsFile = args[6];
        int topK = Integer.parseInt(args[7]);
        String dataType = args[8];
        String outFile = args[9];
        boolean parallel = args[10].equals("true");

        new CreateEntitySimilarityData(paraIndex, entityPassageFile, entityRunFile, entityFile, queryIdToNameFile,
                entityIdToNameFile, stopWordsFile, topK, dataType, outFile, parallel);


    }
}


