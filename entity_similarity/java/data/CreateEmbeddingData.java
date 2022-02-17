package data;

import help.LuceneHelper;
import help.RankingHelper;
import help.Utilities;
import json.AspectLinkExample;
import me.tongfei.progressbar.ProgressBar;
import org.apache.lucene.search.IndexSearcher;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.json.JSONException;
import org.json.JSONObject;
import static java.util.Map.entry;

import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Data for creating embeddings from trained model.
 * Data created from CIKM aspect linking data.
 */


public abstract class CreateEmbeddingData {
    protected final List<AspectLinkExample> aspectLinkExamples;
    protected final Map<String, String> entityParaMap; // Map containing (entity_id, List(para_id)) where para_id --> contains link to entity_id
    protected final IndexSearcher paraIndexSearcher;
    protected final IndexSearcher entityIndexSearcher;
    protected final List<String> stopWords;
    protected Map<String, String> entityMap; // Map containing (ContextId --> List[EntityId]) mappings.
    protected int total = 0;
    protected final List<String> dataStrings = new ArrayList<>();
    protected final AtomicInteger count = new AtomicInteger(0);
    protected final boolean parallel;
    
    public CreateEmbeddingData(String paraIndex,
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
        entityMap = Utilities.readTsvFile(entityFile);
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

    protected void getData(@NotNull AspectLinkExample aspectLinkExample) {

        String exampleId = aspectLinkExample.getId();
        String sentenceContext = aspectLinkExample.getContext().getSentenceContext().getContent();

        if (entityMap.containsKey(exampleId)) {
            // Get data for context
            Map<String, String> context = getContextData(exampleId, sentenceContext);

            try {
                // Get context and aspect entities
                List<String> contextEntityList = Utilities.JSONArrayToList(new JSONObject(entityMap.get(aspectLinkExample.getId()))
                        .getJSONArray("context_entities"));
                List<String> aspectEntityList = Utilities.JSONArrayToList(new JSONObject(entityMap.get(aspectLinkExample.getId()))
                        .getJSONArray("aspect_entities"));

                // Get data for context entities
                List<Map<String, Object>> contextEntityData = getEntityData(sentenceContext, contextEntityList);

                // Get data for aspect entities
                List<Map<String, Object>> aspectEntityData = getEntityData(sentenceContext, aspectEntityList);

                // Create data strings
                if (!contextEntityData.isEmpty() && !aspectEntityData.isEmpty()) {
                    dataStrings.add(toJSONString(context, aspectEntityData, contextEntityData));
                }

            } catch (JSONException e) {
                e.printStackTrace();
            }
        }


        if (parallel) {
            count.getAndIncrement();
            System.out.println("Done: " + aspectLinkExample.getId() + " ( " + count + "/" + total + " ).");
        }

    }

    protected Map<String, String> getContextData(String exampleId, String context) {
        return Map.ofEntries(
                entry("id", exampleId),
                entry("text", String.join(
                        " ",
                        RankingHelper.preProcess(
                                context,
                                stopWords
                        )
                ))
        );
    }

    @NotNull
    protected List<Map<String, Object>> getEntityData(String context, @NotNull List<String> entityList) {
        List<Map<String, Object>> data = new ArrayList<>();
        for (String entityId : entityList) {
            Map<String, Object> entityData = getDataForEntity(entityId, context);
            if (entityData != null) {
                data.add(entityData);
            } else {
                System.out.println("No data found for entity: " + entityId);
            }
        }
        return data;
    }

    @Nullable
    protected Map<String, Object> getDataForEntity(@NotNull String entityId, String sentenceContext) {
        Map<String, Object> doc = new HashMap<>();
        String entityName = Utilities.getEntityName(entityId, entityIndexSearcher);
        String entityDesc = getEntityDescription(sentenceContext, entityId, entityName);
        if (entityDesc != null) {
            doc.put("entity_name", String.join(
                    " ",
                    RankingHelper.preProcess(
                            entityName,
                            stopWords
                    )
            ));
            doc.put("entity_desc", String.join(
                    " ",
                    RankingHelper.preProcess(
                            entityDesc,
                            stopWords
                    )
            ));
            doc.put("entity_types", Utilities.getEntityCategories(entityId, entityIndexSearcher, stopWords));
            doc.put("entity_id", entityId);
            return doc;
        } else {
            return null;
        }
    }

    protected String toJSONString(Map<String, String> context,
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

    protected abstract String getEntityDescription(String sentenceContext, String entityId, String entityName);



}
