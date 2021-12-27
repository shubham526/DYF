package json;

import org.jetbrains.annotations.NotNull;
import org.json.JSONException;
import org.json.JSONObject;

public class Context {
    private final String targetEntity;
    private final Location location;
    private final AnnotatedText sentenceContext;
    private final AnnotatedText paraContext;

    public Context(@NotNull JSONObject jsonObject) throws JSONException {
        this.targetEntity = jsonObject.getString("target_entity");
        this.location = new Location(jsonObject.getJSONObject("location"));
        this.sentenceContext = new AnnotatedText(jsonObject.getJSONObject("sentence"));
        this.paraContext = new AnnotatedText(jsonObject.getJSONObject("paragraph"));

    }

    @Override
    public String toString() {
        return "Context{" +
                "targetEntity='" + targetEntity + '\'' +
                ", location=" + location +
                ", sentenceContext=" + sentenceContext +
                ", paraContext=" + paraContext +
                '}';
    }

    public String getTargetEntity() {
        return targetEntity;
    }

    public Location getLocation() {
        return location;
    }

    public AnnotatedText getSentenceContext() {
        return sentenceContext;
    }

    public AnnotatedText getParaContext() {
        return paraContext;
    }
}
