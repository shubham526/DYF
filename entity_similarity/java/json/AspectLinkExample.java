package json;

import org.jetbrains.annotations.NotNull;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class AspectLinkExample {
    private final String unHashedId;
    private final String id;
    private final String trueAspect;
    private final Context context;
    private final List<Aspect> candidateAspects = new ArrayList<>();


    public AspectLinkExample(@NotNull JSONObject jsonObject) throws JSONException {
        this.unHashedId = jsonObject.getString("unhashed_id");
        this.id = jsonObject.getString("id");
        this.trueAspect = jsonObject.getString("true_aspect");
        this.context = new Context(jsonObject.getJSONObject("context"));
        JSONArray candidateAspects = jsonObject.getJSONArray("candidate_aspects");

        for (int i = 0; i < candidateAspects.length(); i++) {
            this.candidateAspects.add(new Aspect(candidateAspects.getJSONObject(i)));
        }

    }

    public String getUnHashedId() {
        return unHashedId;
    }

    public String getId() {
        return id;
    }

    public String getTrueAspect() {
        return trueAspect;
    }

    public Context getContext() {
        return context;
    }

    public List<Aspect> getCandidateAspects() {
        return candidateAspects;
    }

    @Override
    public String toString() {
        return "AspectLinkExample{" +
                "unHashedId='" + unHashedId + '\'' +
                ", id='" + id + '\'' +
                ", trueAspect='" + trueAspect + '\'' +
                ", context=" + context +
                ", candidateAspects=" + candidateAspects +
                '}';
    }


}
