// lib/main.dart
// Visora AI UCVE-X Flutter Frontend (single file)
// Features:
// - Script -> Video (select language, short/long, voice choice)
// - Upload user voice sample (multipart POST to backend /upload_voice_sample)
// - Upload character photos (multipart POST to backend /upload_character_photo)
// - Generate character voices (calls /generate_character_voices)
// - Trigger render (calls /pro/render_request or /pro/photo_motion_request)
// - Poll job status and show preview
//
// NOTE:
// - Make sure to add required dependencies in pubspec.yaml:
//   http, image_picker, file_picker, video_player, path, path_provider, permission_handler
//
// - Adjust backendBase if your backend URL is different.

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';
import 'package:video_player/video_player.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';

void main() {
  runApp(const VisoraApp());
}

const String backendBase = "https://visora-ai-5nqs.onrender.com";

class VisoraApp extends StatelessWidget {
  const VisoraApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Visora AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        colorScheme: ColorScheme.dark(primary: Colors.purpleAccent),
        scaffoldBackgroundColor: const Color(0xFF0B0B0F),
      ),
      home: const ModeSelectorScreen(),
    );
  }
}

class ModeSelectorScreen extends StatelessWidget {
  const ModeSelectorScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Visora AI — UCVE-X'),
        centerTitle: true,
        backgroundColor: Colors.black87,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(children: [
          const SizedBox(height: 20),
          const Text('Choose Mode', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const ScriptToVideoScreen())),
            icon: const Icon(Icons.text_snippet),
            label: const Text('Script → Video'),
            style: ElevatedButton.styleFrom(minimumSize: const Size.fromHeight(50)),
          ),
          const SizedBox(height: 16),
          ElevatedButton.icon(
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const ImageToVideoScreen())),
            icon: const Icon(Icons.image),
            label: const Text('Image → Video'),
            style: ElevatedButton.styleFrom(backgroundColor: Colors.teal, minimumSize: const Size.fromHeight(50)),
          ),
          const Spacer(),
          const Text('Made with 💜 by Aimantuvya', style: TextStyle(color: Colors.white38)),
        ]),
      ),
    );
  }
}

// ----------------------------- Script -> Video Screen -----------------------------
class ScriptToVideoScreen extends StatefulWidget {
  const ScriptToVideoScreen({super.key});

  @override
  State<ScriptToVideoScreen> createState() => _ScriptToVideoScreenState();
}

class _ScriptToVideoScreenState extends State<ScriptToVideoScreen> {
  final TextEditingController scriptController = TextEditingController();
  String selectedLanguage = "hi";
  String selectedType = "short"; // short / long
  String voiceMode = "auto"; // ai / user / auto
  bool isLoading = false;

  // uploaded assets keys returned from backend
  String? uploadedVoiceKey;
  Map<String, String> uploadedPhotos = {}; // characterName -> photoPath (server saved path or url)

  // helper to pick audio file and upload as voice sample
  Future<void> pickAndUploadVoice() async {
    try {
      // request storage permission
      await Permission.storage.request();
      final result = await FilePicker.platform.pickFiles(type: FileType.audio);
      if (result == null || result.files.isEmpty) return;
      final file = File(result.files.single.path!);
      // show upload dialog for metadata
      final meta = await showDialog<_VoiceMeta>(
        context: context,
        builder: (_) => VoiceMetaDialog(),
      );
      if (meta == null) return;
      setState(() => isLoading = true);
      final voiceKey = await uploadVoiceSample(file, meta.displayName, meta.lang, meta.gender, meta.ageGroup, meta.consent);
      if (voiceKey != null) {
        setState(() {
          uploadedVoiceKey = voiceKey;
        });
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Voice uploaded successfully")));
      } else {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Voice upload failed")));
      }
    } catch (e) {
      debugPrint("Voice upload error: $e");
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error: $e")));
    } finally {
      setState(() => isLoading = false);
    }
  }

  Future<String?> uploadVoiceSample(File audioFile, String displayName, String lang, String gender, String ageGroup, bool consent) async {
    try {
      final uri = Uri.parse("$backendBase/upload_voice_sample");
      final request = http.MultipartRequest('POST', uri);
      request.fields["owner_id"] = "mobile_user"; // replace with auth user id if available
      request.fields["display_name"] = displayName;
      request.fields["lang"] = lang;
      request.fields["gender"] = gender;
      request.fields["age_group"] = ageGroup;
      request.fields["consent"] = consent ? "true" : "false";
      request.files.add(await http.MultipartFile.fromPath('sample', audioFile.path, filename: p.basename(audioFile.path)));
      final streamed = await request.send();
      final resp = await http.Response.fromStream(streamed);
      if (resp.statusCode == 200) {
        final data = jsonDecode(resp.body);
        return data["voice_key"] ?? data["voice_key"] as String?;
      } else {
        debugPrint("Voice upload failed: ${resp.statusCode} ${resp.body}");
        return null;
      }
    } catch (e) {
      debugPrint("uploadVoiceSample error: $e");
      return null;
    }
  }

  // pick image and upload as character photo
  Future<void> pickAndUploadPhoto() async {
    try {
      await Permission.photos.request();
      final picker = ImagePicker();
      final picked = await picker.pickImage(source: ImageSource.gallery, imageQuality: 85);
      if (picked == null) return;
      // ask for character name
      final charName = await _askCharacterName();
      if (charName == null || charName.trim().isEmpty) return;
      setState(() => isLoading = true);
      final key = await uploadCharacterPhoto(File(picked.path), charName);
      if (key != null) {
        setState(() {
          uploadedPhotos[charName] = key;
        });
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Photo uploaded")));
      } else {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Photo upload failed")));
      }
    } catch (e) {
      debugPrint("Photo upload error: $e");
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error: $e")));
    } finally {
      setState(() => isLoading = false);
    }
  }

  Future<String?> uploadCharacterPhoto(File imageFile, String characterName) async {
    try {
      final uri = Uri.parse("$backendBase/upload_character_photo");
      final request = http.MultipartRequest('POST', uri);
      request.fields["character_name"] = characterName;
      request.files.add(await http.MultipartFile.fromPath('photo', imageFile.path, filename: p.basename(imageFile.path)));
      final streamed = await request.send();
      final resp = await http.Response.fromStream(streamed);
      if (resp.statusCode == 200) {
        final data = jsonDecode(resp.body);
        return data["photo"] ?? data["photo"] as String?;
      } else {
        debugPrint("photo upload failed: ${resp.statusCode} ${resp.body}");
        return null;
      }
    } catch (e) {
      debugPrint("uploadCharacterPhoto error: $e");
      return null;
    }
  }

  Future<String?> _askCharacterName() async {
    String name = "";
    final res = await showDialog<String?>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text("Character name"),
        content: TextField(onChanged: (v) => name = v, decoration: const InputDecoration(hintText: "e.g. Mahesh")),
        actions: [
          TextButton(onPressed: () => Navigator.pop(c, null), child: const Text("Cancel")),
          TextButton(onPressed: () => Navigator.pop(c, name.trim()), child: const Text("OK")),
        ],
      ),
    );
    return res;
  }

  Future<void> generateVideo() async {
    final script = scriptController.text.trim();
    if (script.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Please enter script")));
      return;
    }
    setState(() => isLoading = true);
    try {
      // If user uploaded voice sample, use it; else voiceMode chosen
      final payload = {
        "script_text": script,
        "lang": selectedLanguage,
        "duration_sec": selectedType == "short" ? 45 : 150,
        "voice_mode": voiceMode,
        "user_voice_key": uploadedVoiceKey ?? "",
        "photos": uploadedPhotos, // backend may accept mapping; else ignore
        "user_id": "mobile_user"
      };

      final res = await http.post(
        Uri.parse("$backendBase/pro/render_request"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(payload),
      );
      final data = jsonDecode(res.body);
      if (res.statusCode == 200 && data["job_id"] != null) {
        Navigator.push(context, MaterialPageRoute(builder: (_) => RenderProgressScreen(jobId: data["job_id"])));
      } else {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error: ${data['detail'] ?? data}")));
      }
    } catch (e) {
      debugPrint("generateVideo error: $e");
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Network error: $e")));
    } finally {
      setState(() => isLoading = false);
    }
  }

  @override
  void dispose() {
    scriptController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final photoWidgets = uploadedPhotos.entries.map((e) => Chip(label: Text("${e.key}"))).toList();
    return Scaffold(
      appBar: AppBar(title: const Text("Script → Video")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: ListView(children: [
          const Text("Script (lines: CharacterName: Dialogue)", style: TextStyle(color: Colors.white70)),
          const SizedBox(height: 8),
          TextField(controller: scriptController, maxLines: 8, style: const TextStyle(color: Colors.white)),
          const SizedBox(height: 12),
          Row(children: [
            Expanded(child: DropdownButtonFormField<String>(
              value: selectedLanguage,
              items: const [
                DropdownMenuItem(value: "hi", child: Text("Hindi")),
                DropdownMenuItem(value: "en", child: Text("English")),
                DropdownMenuItem(value: "ta", child: Text("Tamil")),
                DropdownMenuItem(value: "te", child: Text("Telugu")),
              ],
              onChanged: (v) => setState(() => selectedLanguage = v ?? "hi"),
              decoration: const InputDecoration(labelText: "Language"),
            )),
            const SizedBox(width: 10),
            Expanded(child: DropdownButtonFormField<String>(
              value: selectedType,
              items: const [
                DropdownMenuItem(value: "short", child: Text("Short")),
                DropdownMenuItem(value: "long", child: Text("Long")),
              ],
              onChanged: (v) => setState(() => selectedType = v ?? "short"),
              decoration: const InputDecoration(labelText: "Type"),
            )),
          ]),
          const SizedBox(height: 12),
          const Text("Voice Option:", style: TextStyle(color: Colors.white70)),
          Wrap(spacing: 8, children: [
            ChoiceChip(label: const Text("AI"), selected: voiceMode == "ai", onSelected: (_) => setState(() => voiceMode = "ai")),
            ChoiceChip(label: const Text("User (Upload)"), selected: voiceMode == "user", onSelected: (_) => setState(() => voiceMode = "user")),
            ChoiceChip(label: const Text("Auto"), selected: voiceMode == "auto", onSelected: (_) => setState(() => voiceMode = "auto")),
          ]),
          const SizedBox(height: 8),
          Row(children: [
            ElevatedButton.icon(
              onPressed: isLoading ? null : pickAndUploadVoice,
              icon: const Icon(Icons.mic),
              label: Text(uploadedVoiceKey == null ? "Upload Voice Sample" : "Voice Uploaded"),
            ),
            const SizedBox(width: 12),
            ElevatedButton.icon(
              onPressed: isLoading ? null : pickAndUploadPhoto,
              icon: const Icon(Icons.person),
              label: const Text("Upload Character Photo"),
            ),
          ]),
          const SizedBox(height: 8),
          Wrap(children: photoWidgets),
          const SizedBox(height: 16),
          ElevatedButton.icon(
            onPressed: isLoading ? null : generateVideo,
            icon: isLoading ? const CircularProgressIndicator() : const Icon(Icons.play_arrow),
            label: Text(isLoading ? "Processing..." : "Generate Video"),
            style: ElevatedButton.styleFrom(minimumSize: const Size.fromHeight(48)),
          ),
        ]),
      ),
    );
  }
}

// ----------------------------- Image -> Video Screen -----------------------------
class ImageToVideoScreen extends StatefulWidget {
  const ImageToVideoScreen({super.key});

  @override
  State<ImageToVideoScreen> createState() => _ImageToVideoScreenState();
}

class _ImageToVideoScreenState extends State<ImageToVideoScreen> {
  final ImagePicker _picker = ImagePicker();
  List<XFile> imgs = [];
  bool isLoading = false;

  Future<void> pickImages() async {
    try {
      await Permission.photos.request();
      final picked = await _picker.pickMultiImage(imageQuality: 85);
      if (picked != null && picked.isNotEmpty) {
        setState(() => imgs = picked);
      }
    } catch (e) {
      debugPrint("pickImages error: $e");
    }
  }

  Future<void> uploadImagesAndGenerate() async {
    if (imgs.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Please select images")));
      return;
    }
    setState(() => isLoading = true);
    try {
      // upload images one by one to backend /upload_character_photo with autogenerated names
      Map<String, String> photoMap = {};
      for (var f in imgs) {
        final name = p.basename(f.path);
        // use char name as "img1","img2" unless user sets
        final charName = "img_${photoMap.length + 1}";
        final key = await uploadCharacterPhoto(File(f.path), charName);
        if (key != null) photoMap[charName] = key;
      }
      // call photo_motion_request with photoMap
      final payload = {
        "images": photoMap.values.toList(),
        "lang": "hi",
        "duration_sec": 30,
        "voice_mode": "ai",
        "user_id": "mobile_user"
      };
      final res = await http.post(Uri.parse("$backendBase/pro/photo_motion_request"), headers: {"Content-Type": "application/json"}, body: jsonEncode(payload));
      final data = jsonDecode(res.body);
      if (res.statusCode == 200 && data["job_id"] != null) {
        Navigator.push(context, MaterialPageRoute(builder: (_) => RenderProgressScreen(jobId: data["job_id"])));
      } else {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error: ${data}")));
      }
    } catch (e) {
      debugPrint("uploadImagesAndGenerate error: $e");
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error: $e")));
    } finally {
      setState(() => isLoading = false);
    }
  }

  // reuse uploadCharacterPhoto from previous screen; duplicate for scope
  Future<String?> uploadCharacterPhoto(File imageFile, String characterName) async {
    try {
      final uri = Uri.parse("$backendBase/upload_character_photo");
      final request = http.MultipartRequest('POST', uri);
      request.fields["character_name"] = characterName;
      request.files.add(await http.MultipartFile.fromPath('photo', imageFile.path, filename: p.basename(imageFile.path)));
      final streamed = await request.send();
      final resp = await http.Response.fromStream(streamed);
      if (resp.statusCode == 200) {
        final data = jsonDecode(resp.body);
        return data["photo"] ?? data["photo"] as String?;
      } else {
        debugPrint("photo upload failed: ${resp.statusCode} ${resp.body}");
        return null;
      }
    } catch (e) {
      debugPrint("uploadCharacterPhoto error: $e");
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    final thumbs = imgs.map((e) => Padding(
      padding: const EdgeInsets.all(6.0),
      child: Image.file(File(e.path), width: 100, height: 100, fit: BoxFit.cover),
    )).toList();
    return Scaffold(
      appBar: AppBar(title: const Text("Image → Video")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(children: [
          ElevatedButton.icon(onPressed: pickImages, icon: const Icon(Icons.photo_library), label: const Text("Pick Images")),
          const SizedBox(height: 8),
          SingleChildScrollView(scrollDirection: Axis.horizontal, child: Row(children: thumbs)),
          const Spacer(),
          ElevatedButton.icon(
            onPressed: isLoading ? null : uploadImagesAndGenerate,
            icon: isLoading ? const CircularProgressIndicator() : const Icon(Icons.play_arrow),
            label: const Text("Generate Video from Images"),
            style: ElevatedButton.styleFrom(minimumSize: const Size.fromHeight(48)),
          ),
        ]),
      ),
    );
  }
}

// ----------------------------- Render Progress Screen -----------------------------
class RenderProgressScreen extends StatefulWidget {
  final String jobId;
  const RenderProgressScreen({super.key, required this.jobId});

  @override
  State<RenderProgressScreen> createState() => _RenderProgressScreenState();
}

class _RenderProgressScreenState extends State<RenderProgressScreen> {
  String status = "queued";
  double progress = 0.0;
  String? finalVideoUrl;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    pollStatus();
    _timer = Timer.periodic(const Duration(seconds: 5), (_) => pollStatus());
  }

  Future<void> pollStatus() async {
    try {
      final res = await http.get(Uri.parse("$backendBase/pro/job_status/${widget.jobId}"));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        setState(() {
          status = data["status"] ?? status;
          progress = (data["progress"] != null) ? (data["progress"] as num).toDouble() : progress;
          finalVideoUrl = data["video"] ?? data["final_video"] ?? finalVideoUrl;
        });
        if (status == "done" || status == "failed" || status == "cancelled") {
          _timer?.cancel();
          if (status == "done" && finalVideoUrl != null) {
            // show preview
            Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => PreviewScreen(videoUrl: finalVideoUrl!, jobId: widget.jobId)));
          }
        }
      } else {
        debugPrint("status poll error: ${res.statusCode}");
      }
    } catch (e) {
      debugPrint("pollStatus error: $e");
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Widget statusWidget() {
    if (status == "done") return const Text("Done", style: TextStyle(color: Colors.green));
    if (status == "failed") return const Text("Failed", style: TextStyle(color: Colors.red));
    return Column(children: [
      Text(status, style: const TextStyle(fontSize: 18)),
      const SizedBox(height: 8),
      LinearProgressIndicator(value: progress, color: Colors.purpleAccent),
    ]);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Render Status")),
      body: Center(child: Padding(padding: const EdgeInsets.all(20), child: statusWidget())),
    );
  }
}

// ----------------------------- Preview Screen (play and upload) -----------------------------
class PreviewScreen extends StatefulWidget {
  final String videoUrl;
  final String jobId;
  const PreviewScreen({super.key, required this.videoUrl, required this.jobId});

  @override
  State<PreviewScreen> createState() => _PreviewScreenState();
}

class _PreviewScreenState extends State<PreviewScreen> {
  VideoPlayerController? _controller;
  bool isUploading = false;
  final TextEditingController titleController = TextEditingController();
  final TextEditingController descController = TextEditingController();
  final TextEditingController tagsController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _controller = VideoPlayerController.network(widget.videoUrl)
      ..initialize().then((_) {
        setState(() {});
        _controller?.play();
      });
    // auto metadata defaults
    titleController.text = "Visora AI Video";
    descController.text = "Created with Visora AI UCVE-X";
    tagsController.text = "#VisoraAI #AI";
  }

  Future<void> confirmUpload(String platform, String platformId) async {
    setState(() => isUploading = true);
    try {
      final res = await http.post(Uri.parse("$backendBase/publish_to_social"), headers: {"Content-Type": "application/json"}, body: jsonEncode({
        "job_id": widget.jobId,
        "youtube_id": platform == "youtube" ? platformId : null,
        "instagram_id": platform == "instagram" ? platformId : null,
        "facebook_id": platform == "facebook" ? platformId : null,
        "quote_mode": false,
      }));
      final data = jsonDecode(res.body);
      if (res.statusCode == 200) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Upload simulated / success")));
      } else {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Upload error: ${data}")));
      }
    } catch (e) {
      debugPrint("confirmUpload error: $e");
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error: $e")));
    } finally {
      setState(() => isUploading = false);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    titleController.dispose();
    descController.dispose();
    tagsController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final videoReady = _controller != null && _controller!.value.isInitialized;
    return Scaffold(
      appBar: AppBar(title: const Text("Preview & Upload")),
      body: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(children: [
          if (videoReady)
            AspectRatio(aspectRatio: _controller!.value.aspectRatio, child: VideoPlayer(_controller!))
          else
            Container(height: 200, color: Colors.black12, child: const Center(child: CircularProgressIndicator())),
          const SizedBox(height: 8),
          TextField(controller: titleController, decoration: const InputDecoration(labelText: "Title")),
          const SizedBox(height: 8),
          TextField(controller: descController, maxLines: 2, decoration: const InputDecoration(labelText: "Description")),
          const SizedBox(height: 8),
          TextField(controller: tagsController, decoration: const InputDecoration(labelText: "Tags (comma)")),
          const SizedBox(height: 12),
          Row(children: [
            Expanded(child: ElevatedButton(onPressed: isUploading ? null : () => confirmUpload("youtube", "UC_YOUR_ID"), child: const Text("Upload to YouTube"))),
            const SizedBox(width: 8),
            Expanded(child: ElevatedButton(onPressed: isUploading ? null : () => confirmUpload("instagram", "insta_id"), child: const Text("Instagram"))),
          ]),
          const SizedBox(height: 8),
          Row(children: [
            Expanded(child: ElevatedButton(onPressed: isUploading ? null : () => confirmUpload("facebook", "fb_id"), child: const Text("Facebook"))),
          ]),
          const SizedBox(height: 8),
          if (isUploading) const LinearProgressIndicator(),
        ]),
      ),
    );
  }
}

// ----------------------------- Small Dialogs & Helpers -----------------------------
class _VoiceMeta {
  final String displayName;
  final String lang;
  final String gender;
  final String ageGroup;
  final bool consent;
  _VoiceMeta(this.displayName, this.lang, this.gender, this.ageGroup, this.consent);
}

class VoiceMetaDialog extends StatefulWidget {
  @override
  State<VoiceMetaDialog> createState() => _VoiceMetaDialogState();
}

class _VoiceMetaDialogState extends State<VoiceMetaDialog> {
  final _nameCtrl = TextEditingController();
  String lang = "hi";
  String gender = "male";
  String ageGroup = "adult";
  bool consent = false;

  @override
  void dispose() {
    _nameCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text("Voice Sample Metadata"),
      content: SingleChildScrollView(
        child: Column(children: [
          TextField(controller: _nameCtrl, decoration: const InputDecoration(labelText: "Display name (e.g. Mahesh)")),
          DropdownButtonFormField<String>(value: lang, items: const [
            DropdownMenuItem(value: "hi", child: Text("Hindi")),
            DropdownMenuItem(value: "en", child: Text("English")),
            DropdownMenuItem(value: "ta", child: Text("Tamil")),
          ], onChanged: (v) => setState(() => lang = v ?? "hi")),
          DropdownButtonFormField<String>(value: gender, items: const [
            DropdownMenuItem(value: "male", child: Text("Male")),
            DropdownMenuItem(value: "female", child: Text("Female")),
            DropdownMenuItem(value: "child", child: Text("Child")),
          ], onChanged: (v) => setState(() => gender = v ?? "male")),
          DropdownButtonFormField<String>(value: ageGroup, items: const [
            DropdownMenuItem(value: "adult", child: Text("Adult")),
            DropdownMenuItem(value: "old", child: Text("Old")),
          ], onChanged: (v) => setState(() => ageGroup = v ?? "adult")),
          CheckboxListTile(value: consent, onChanged: (v) => setState(() => consent = v ?? false), title: const Text("I give consent to use this voice sample")),
        ]),
      ),
      actions: [
        TextButton(onPressed: () => Navigator.pop(context, null), child: const Text("Cancel")),
        TextButton(onPressed: () {
          final name = _nameCtrl.text.trim();
          if (name.isEmpty) return;
          Navigator.pop(context, _VoiceMeta(name, lang, gender, ageGroup, consent));
        }, child: const Text("OK")),
      ],
    );
  }
}
