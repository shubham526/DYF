import help.LuceneHelper;
import help.RankingHelper;
import help.Utilities;
import help.Utilities.EntityContextDocument;
import json.Aspect;
import json.AspectLinkExample;
import json.Entity;
import me.tongfei.progressbar.ProgressBar;
import org.apache.lucene.document.Document;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import static java.util.Map.entry;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Data for creating embeddings from trained model.
 * Data created from CIKM aspect linking data.
 */


public class CreateEmbeddingData {
    private final List<AspectLinkExample> aspectLinkExamples;
    private final Map<String, String> entityParaMap; // Map containing (entity_id, List(para_id)) where para_id --> contains link to entity_id
    private final IndexSearcher paraIndexSearcher;
    private final IndexSearcher entityIndexSearcher;
    private final List<String> stopWords;
    private int total = 0;
    private final List<String> dataStrings = new ArrayList<>();
    private final AtomicInteger count = new AtomicInteger(0);
    private final boolean parallel;
    
    CreateEmbeddingData(String paraIndex,
                        String entityIndex,
                        String entityParaFile,
                        String stopWordsFile,
                        String dataFile,
                        String entityFile,
                        boolean parallel) {
        
        this.parallel = parallel;

        System.out.print("Setting up paragraph index...");
        this.paraIndexSearcher = LuceneHelper.createSearcher(paraIndex, "bm25");
        System.out.println("[Done].");

        System.out.print("Setting up entity index...");
        this.entityIndexSearcher = LuceneHelper.createSearcher(entityIndex, "bm25");
        System.out.println("[Done].");

        System.out.print("Loading entity to passage mappings...");
        entityParaMap = Utilities.readTsvFile(entityParaFile);
        System.out.println("[Done].");

        System.out.print("Loading stop words....");
        stopWords = Utilities.getStopWords(stopWordsFile);
        System.out.println("[Done].");

        System.out.print("Loading data...");
        aspectLinkExamples = Utilities.readJSONLFile(dataFile);
        System.out.println("[Done].");

        System.out.print("Loading entities file...");
        Map<String, String> entityMap = Utilities.readTsvFile(entityFile);
        System.out.println("[Done].");
    }


    public void doTask(String outFile) {
        total = aspectLinkExamples.size();

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
            aspectLinkExamples.parallelStream().forEach(this::getData);
        } else {
            System.out.println("Using Sequential Streams.");

            // Do in serial
            ProgressBar pb = new ProgressBar("Progress", aspectLinkExamples.size());
            for (AspectLinkExample aspectLinkExample: aspectLinkExamples) {
                getData(aspectLinkExample);
                pb.step();
            }
            pb.close();
        }
        System.out.print("Writing to file....");
        Utilities.writeToFile(outFile, dataStrings);
        System.out.println("[Done].");
    }

    private void getData(@NotNull AspectLinkExample aspectLinkExample) {

        Map<String, String> context = Map.ofEntries(
                entry("id", aspectLinkExample.getId()),
                entry("text", String.join(
                        " ",
                        RankingHelper.preProcess(
                                aspectLinkExample.getContext().getSentenceContext().getContent(),
                                stopWords
                        )
                ))
        );
        List<Map<String, Object>> aspectEntityData = getAspectEntityData(aspectLinkExample);
        List<Map<String, Object>> contextEntityData = getContextEntityData(aspectLinkExample);

        if (!aspectEntityData.isEmpty() && !contextEntityData.isEmpty()) {
            dataStrings.add(toJSONString(context, aspectEntityData, contextEntityData));
        } else {
            if (aspectEntityData.isEmpty()) {
                System.err.println("No aspect entities found. Skipping: " +  aspectLinkExample.getId());
            } else  {
                System.err.println("No context entities found. Skipping: " +  aspectLinkExample.getId());
            }
        }

        if (parallel) {
            count.getAndIncrement();
            System.out.println("Done: " + aspectLinkExample.getId() + " ( " + count + "/" + total + " ).");
        }


    }

    private String toJSONString(Map<String, String> context,
                                List<Map<String, Object>> aspectEntityData,
                                List<Map<String, Object>> contextEntityData) {
        JSONObject example = new JSONObject();
        try {
            example.put("context", context);
            example.put("context_entities", aspectEntityData);
            example.put("aspect_entities", contextEntityData);
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return example.toString();
    }

    @NotNull
    private List<Map<String, Object>> getContextEntityData(@NotNull AspectLinkExample aspectLinkExample) {
        List<Map<String, Object>> data = new ArrayList<>();

        String sentenceContext = aspectLinkExample.getContext().getSentenceContext().getContent();
        List<Entity> contextEntityList = aspectLinkExample.getContext().getParaContext().getEntities();


        for (Entity entity : contextEntityList) {
            Map<String, Object> entityData = getDataForEntity(entity, sentenceContext);
            if (entityData != null) {
                data.add(entityData);
            } else {
                System.out.println("ContextId: " + aspectLinkExample.getId() + " " + "No data for context entity: " + entity.getEntityId());
            }
        }
        return data;
    }

    @NotNull
    private List<Map<String, Object>> getAspectEntityData(@NotNull AspectLinkExample aspectLinkExample) {

        List<Map<String, Object>> data = new ArrayList<>();

        String sentenceContext = aspectLinkExample.getContext().getSentenceContext().getContent();

        // Get all entities from all aspects
        List<Entity> aspectEntityList = getAllEntitiesFromAspects(aspectLinkExample.getCandidateAspects());

        for (Entity entity : aspectEntityList) {
            //if (!entity.getEntityId().equals("enwiki:Kapilavastu%20(ancient%20city)")) continue;
            Map<String, Object> entityData = getDataForEntity(entity, sentenceContext);
            if (entityData != null) {
                data.add(entityData);
            } else {
                System.out.println("ContextId: " + aspectLinkExample.getId() + " " + "No data for aspect entity: " + entity.getEntityId());
            }

        }
        return data;
    }


    @Nullable
    private Map<String, Object> getDataForEntity(@NotNull Entity entity, String sentenceContext) {
        Map<String, Object> doc = new HashMap<>();
        String supportPsg = getSupportPsgForEntity(sentenceContext, entity);
        if (supportPsg != null) {
            doc.put("entity_name", String.join(
                    " ",
                    RankingHelper.preProcess(
                            entity.getEntityName(),
                            stopWords
                    )
            ));
            doc.put("entity_desc", String.join(
                    " ",
                    RankingHelper.preProcess(
                            supportPsg,
                            stopWords
                    )
            ));
            doc.put("entity_types", Utilities.getEntityCategories(entity.getEntityId(), entityIndexSearcher, stopWords));
            doc.put("entity_id", entity.getEntityId());
            return doc;
        } else {
            return null;
        }
    }

    @Nullable
    private String getSupportPsgForEntity(String sentenceContext, @NotNull Entity entity) {
        try {
            // Get the paragraphs which mention the entity
            String entityId = entity.getEntityId();
            String entityName = entity.getEntityName();
            if (entityParaMap.containsKey(entityId)) {
                List<String> paraList = JSONArrayToList(new JSONObject(entityParaMap.get(entityId))
                        .getJSONArray("paragraphs"));

                // Rank these paragraphs for the query
                List<RankingHelper.ScoredDocument> rankedParaList = rankParasForQuery(sentenceContext, entityName, paraList);

                if (!rankedParaList.isEmpty()) {

                    // Create the ECD using the ranked paragraphs
                    EntityContextDocument d = createECD(entityId, rankedParaList);
                    if (d != null) {
                        List<String> contextEntityList = d.getEntityList();
                        Map<String, Integer> freqDist = getDistribution(contextEntityList);
                        freqDist.remove(entityId);
                        return getSupportPsgForEntity(d, freqDist);
                    }
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
    protected String getSupportPsgForEntity(@NotNull EntityContextDocument d, Map<String, Integer> freqMap) {

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

    @NotNull
    private Map<String, Integer> getDistribution(@NotNull List<String> contextEntityList) {
        HashMap<String, Integer> freqMap = new HashMap<>();

        // For every co-occurring entity do
        for (String entityID : contextEntityList) {
            freqMap.compute(entityID, (t, oldV) -> (oldV == null) ? 1 : oldV + 1);
        }
        return  Utilities.sortByValueDescending(freqMap);
    }

    @NotNull
    private List<RankingHelper.ScoredDocument> rankParasForQuery(String sentenceContext,
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

    @NotNull
    private List<Entity> getAllEntitiesFromAspects(@NotNull List<Aspect> candidateAspects) {
        List<Entity> entityList = new ArrayList<>();
        Set<String> seen = new HashSet<>();

        for (Aspect candidateAspect : candidateAspects) {
            List<Entity> aspectEntityList = candidateAspect.getAspectContent().getEntities();
            for (Entity aspectEntity : aspectEntityList) {
                if (!seen.contains(aspectEntity.getEntityId())) {
                    entityList.add(aspectEntity);
                    seen.add(aspectEntity.getEntityId());
                }
            }
        }

        return entityList;
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

    public static void main(@NotNull String[] args) {
        String paraIndex = args[0];
        String entityIndex = args[1];
        String entityParaFile = args[2];
        String stopWordsFile = args[3];
        String dataFile = args[4];
        String entityFile = args[5];
        String outFile = args[6];
        boolean parallel = args[7].equals("true");

        CreateEmbeddingData ob = new CreateEmbeddingData(
                paraIndex, entityIndex, entityParaFile, stopWordsFile, dataFile, entityFile, parallel
        );

        ob.doTask(outFile);
    }

}

