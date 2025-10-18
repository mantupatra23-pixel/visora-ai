// lib/screens/editor_screen.dart
// Visora — Video Editor Screen with Admin-controlled pricing (single-file)
// NOTE: This file uses placeholders for video preview and simulated backend.
// Replace preview area with `video_player` and hook _sendEditJobToBackend for real rendering.

import 'dart:convert';
import 'dart:math';
import 'dart:ui';
import 'package:blur/blur.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:lottie/lottie.dart';

class EditorScreen extends StatefulWidget {
  final int initialCredits;
  final String userPlan; // 'Free' / 'Premium' / 'Enterprise'

  const EditorScreen({Key? key, this.initialCredits = 100, this.userPlan = 'Free'}) : super(key: key);

  @override
  State<EditorScreen> createState() => _EditorScreenState();
}

class _EditorScreenState extends State<EditorScreen> with TickerProviderStateMixin {
  // --- Admin controlled pricing JSON (default values) ---
  // Admin can edit this via Admin panel (copy/paste JSON)
  Map<String, dynamic> _pricing = {
    "trim": {"credits": 1, "price": 5},
    "voice_replace": {"credits": 3, "price": 15},
    "filters": {"credits": 2, "price": 10},
    "background_change": {"credits": 4, "price": 20},
    "music_replace": {"credits": 2, "price": 10},
    "auto_enhance": {"credits": 3, "price": 15},
    "subtitle_edit": {"credits": 1, "price": 5},
    "aspect_change": {"credits": 1, "price": 5},
    "re_render": {"credits_per_level": 2, "price_per_level": 10},
    "ai_rebuild": {"credits": 5, "price": 25}
  };

  // --- local editing state ---
  int _userCredits = 0;
  Map<String, dynamic> _selectedEdits = {}; // feature -> params
  int _totalCreditsNeeded = 0;
  double _totalPriceNeeded = 0.0;

  // fake video metadata
  String _videoTitle = "My Project #1";
  Duration _videoDuration = const Duration(seconds: 45);
  String _videoQuality = '1080p';

  // animations
  late AnimationController _bgAnim;
  late Animation<double> _bgAnimVal;

  // UI controllers
  bool _isAdminOpen = false;
  final TextEditingController _adminJsonController = TextEditingController();
  bool _isSubmitting = false;

  @override
  void initState() {
    super.initState();
    _userCredits = widget.initialCredits;
    _bgAnim = AnimationController(vsync: this, duration: const Duration(seconds: 14))..repeat(reverse: true);
    _bgAnimVal = Tween<double>(begin: -0.2, end: 0.2).animate(CurvedAnimation(parent: _bgAnim, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _bgAnim.dispose();
    _adminJsonController.dispose();
    super.dispose();
  }

  // ----------------- pricing logic -----------------
  void _recalculateCost() {
    int credits = 0;
    double price = 0.0;

    // iterate selected edits
    _selectedEdits.forEach((feature, params) {
      if (!_pricing.containsKey(feature)) return;
      final cfg = _pricing[feature];
      if (feature == 're_render') {
        // params: levels (int)
        final levels = (params is Map && params['levels'] is int) ? params['levels'] as int : 1;
        final creditsPerLevel = cfg['credits_per_level'] ?? 0;
        final pricePerLevel = cfg['price_per_level'] ?? cfg['price'] ?? 0;
        credits += creditsPerLevel * levels;
        price += (pricePerLevel * levels);
      } else {
        final c = (cfg['credits'] is int) ? cfg['credits'] as int : int.tryParse(cfg['credits']?.toString() ?? '0') ?? 0;
        final p = (cfg['price'] is num) ? (cfg['price'] as num).toDouble() : double.tryParse(cfg['price']?.toString() ?? '0') ?? 0.0;
        credits += c;
        price += p;
      }
    });

    // apply plan discounts (admin can change logic server-side later)
    double discount = 0.0;
    if (widget.userPlan.toLowerCase() == 'premium') discount = 0.20; // 20% discount
    if (widget.userPlan.toLowerCase() == 'enterprise') discount = 1.0; // free (example)
    final discountedPrice = price * (1 - discount);

    setState(() {
      _totalCreditsNeeded = credits;
      _totalPriceNeeded = double.parse(discountedPrice.toStringAsFixed(2));
    });
  }

  // Admin opens editor JSON editor (no persistence — copy/paste admin JSON)
  void _openAdminPanel() {
    _adminJsonController.text = JsonEncoder.withIndent('  ').convert(_pricing);
    setState(() => _isAdminOpen = true);
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => DraggableScrollableSheet(
        initialChildSize: 0.9,
        minChildSize: 0.5,
        maxChildSize: 0.95,
        builder: (context, sc) {
          return Blur(
            blur: 8,
            borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
            child: Container(
              decoration: BoxDecoration(color: Color(0xFF0F0F12), borderRadius: BorderRadius.vertical(top: Radius.circular(18))),
              padding: const EdgeInsets.all(12),
              child: ListView(
                controller: sc,
                children: [
                  Text('Admin: Edit Pricing JSON', style: GoogleFonts.poppins(fontSize: 18, fontWeight: FontWeight.w700)),
                  const SizedBox(height: 8),
                  Text('Edit pricing structure below (JSON). After Apply, UI will use new prices.', style: TextStyle(color: Colors.white70)),
                  const SizedBox(height: 12),
                  SizedBox(
                    height: 420,
                    child: TextField(
                      controller: _adminJsonController,
                      keyboardType: TextInputType.multiline,
                      maxLines: null,
                      style: const TextStyle(color: Colors.white, fontFamily: 'monospace', fontSize: 12),
                      decoration: InputDecoration(
                        hintText: '{ "trim": {"credits":1,"price":5}, ... }',
                        hintStyle: TextStyle(color: Colors.white30),
                        filled: true,
                        fillColor: Colors.white10,
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(8), borderSide: BorderSide.none),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      ElevatedButton(
                        onPressed: () {
                          // apply JSON
                          try {
                            final Map<String, dynamic> parsed = json.decode(_adminJsonController.text) as Map<String, dynamic>;
                            setState(() {
                              _pricing = parsed;
                              _isAdminOpen = false;
                            });
                            Navigator.of(context).pop();
                            _recalculateCost();
                            ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Pricing updated (applied).')));
                          } catch (e) {
                            ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('JSON parse error: $e')));
                          }
                        },
                        child: const Text('Apply'),
                      ),
                      const SizedBox(width: 8),
                      OutlinedButton(
                        onPressed: () {
                          // export to clipboard
                          final txt = JsonEncoder.withIndent('  ').convert(_pricing);
                          // copying programmatically needs Clipboard; use snackbox showing string length
                          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Pricing JSON length ${txt.length} (copy manually)')));
                        },
                        child: const Text('Export (show length)'),
                      ),
                      const SizedBox(width: 8),
                      TextButton(
                        onPressed: () => Navigator.of(context).pop(),
                        child: const Text('Close'),
                      )
                    ],
                  )
                ],
              ),
            ),
          );
        },
      ),
    ).whenComplete(() {
      setState(() => _isAdminOpen = false);
    });
  }

  // toggle selecting a feature
  void _toggleFeature(String feature, [dynamic params]) {
    setState(() {
      if (_selectedEdits.containsKey(feature)) {
        _selectedEdits.remove(feature);
      } else {
        _selectedEdits[feature] = params ?? true;
      }
      _recalculateCost();
    });
  }

  // special setter for re-render levels
  void _setReRenderLevels(int levels) {
    setState(() {
      if (levels <= 0) {
        _selectedEdits.remove('re_render');
      } else {
        _selectedEdits['re_render'] = {'levels': levels};
      }
      _recalculateCost();
    });
  }

  // preview action (simulate)
  Future<void> _previewChanges() async {
    // simply show a modal with simulated preview and details
    await showDialog(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: Color(0xFF0B0B0D),
        title: Text('Preview Changes', style: TextStyle(color: Colors.white)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('Selected edits:', style: TextStyle(color: Colors.white70)),
            const SizedBox(height: 8),
            ..._selectedEdits.keys.map((k) => Text('- $k', style: TextStyle(color: Colors.white))),
            const SizedBox(height: 12),
            Text('Estimated credits: $_totalCreditsNeeded', style: TextStyle(color: Colors.white)),
            Text('Estimated price: ₹$_totalPriceNeeded', style: TextStyle(color: Colors.white)),
            const SizedBox(height: 14),
            Container(
              height: 120,
              decoration: BoxDecoration(color: Colors.black, borderRadius: BorderRadius.circular(8)),
              child: Center(child: Icon(Icons.play_circle_outline, color: Colors.white, size: 48)),
            ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Close')),
          ElevatedButton(
            onPressed: () {
              Navigator.of(context).pop();
              _confirmAndSubmitJob();
            },
            child: const Text('Confirm & Submit'),
          )
        ],
      ),
    );
  }

  // confirm and submit (deduct credits or show pay)
  Future<void> _confirmAndSubmitJob() async {
    if (_totalCreditsNeeded <= 0) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Koi edit select nahi hua.')));
      return;
    }

    // check credits
    if (_userCredits < _totalCreditsNeeded && widget.userPlan.toLowerCase() != 'enterprise') {
      // insufficient — show pay dialog
      final shortfall = _totalCreditsNeeded - _userCredits;
      await showDialog(
        context: context,
        builder: (_) => AlertDialog(
          backgroundColor: Color(0xFF0B0B0D),
          title: Text('Insufficient Credits', style: TextStyle(color: Colors.white)),
          content: Text('Aapko $shortfall aur credits chahiye. Purchase karein ya plan upgrade karein.', style: TextStyle(color: Colors.white70)),
          actions: [
            TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Cancel')),
            ElevatedButton(
              onPressed: () {
                // simulate purchase (add credits)
                setState(() {
                  _userCredits += shortfall + 10; // give a little extra
                });
                Navigator.of(context).pop();
                ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Purchased $shortfall+10 credits (simulated).')));
              },
              child: const Text('Buy Credits'),
            ),
          ],
        ),
      );
      return;
    }

    // deduct credits (if not enterprise)
    if (widget.userPlan.toLowerCase() != 'enterprise') {
      setState(() {
        _userCredits -= _totalCreditsNeeded;
      });
    }

    // simulate submitting to backend
    setState(() => _isSubmitting = true);
    try {
      await Future.delayed(Duration(seconds: 3 + Random().nextInt(3))); // simulate processing time
      // hook: _sendEditJobToBackend(_selectedEdits, _totalPriceNeeded);
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Edit job submitted. Credits deducted: $_totalCreditsNeeded')));
      // reset selections
      setState(() {
        _selectedEdits.clear();
        _totalCreditsNeeded = 0;
        _totalPriceNeeded = 0.0;
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Submission failed: $e')));
    } finally {
      setState(() => _isSubmitting = false);
    }
  }

  // placeholder for backend submission
  Future<void> _sendEditJobToBackend(Map<String, dynamic> edits, double price) async {
    // TODO: replace with HTTP call passing edits, price, user id, video id etc.
    // Example: POST /api/render/edit { videoId, edits, price, creditsUsed, userId }
    await Future.delayed(Duration(seconds: 1));
    return;
  }

  // ---------------- UI ----------------
  Widget _animatedBackground() {
    return AnimatedBuilder(
      animation: _bgAnimVal,
      builder: (context, child) {
        final v = _bgAnimVal.value;
        return Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment(-1 + v, -0.8 - v),
              end: Alignment(1 - v, 0.8 + v),
              colors: [Color(0xFF0E0B2D), Color(0xFF3A0F5A), Color(0xFFFF5E62)],
            ),
          ),
        );
      },
    );
  }

  Widget _topBar() {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12.0, vertical: 8),
        child: Row(
          children: [
            Text('Editor', style: GoogleFonts.poppins(fontSize: 20, fontWeight: FontWeight.w700, color: Colors.white)),
            const Spacer(),
            Text('Credits: $_userCredits', style: const TextStyle(color: Colors.white70)),
            const SizedBox(width: 12),
            IconButton(
              onPressed: _openAdminPanel,
              icon: const Icon(Icons.admin_panel_settings_outlined, color: Colors.white70),
              tooltip: 'Admin: Edit pricing',
            ),
          ],
        ),
      ),
    );
  }

  Widget _videoPreviewCard() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12.0, vertical: 12),
      child: Blur(
        blur: 6,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          height: 220,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.03),
            borderRadius: BorderRadius.circular(16),
          ),
          child: Stack(
            children: [
              // placeholder thumbnail / video area
              Positioned.fill(
                child: Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(16),
                    image: DecorationImage(image: NetworkImage('https://picsum.photos/seed/editor/800/450'), fit: BoxFit.cover, colorFilter: ColorFilter.mode(Colors.black26, BlendMode.darken)),
                  ),
                ),
              ),
              Positioned(
                left: 14,
                top: 14,
                child: Chip(
                  backgroundColor: Colors.white12,
                  label: Text('Duration: ${_videoDuration.inSeconds}s • $_videoQuality', style: TextStyle(color: Colors.white70)),
                ),
              ),
              Center(child: Icon(Icons.play_circle_outline, color: Colors.white54, size: 80)),
              // floating small badge for selected edits
              Positioned(
                right: 12,
                bottom: 12,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                  decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(12)),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text('${_selectedEdits.length} edits', style: TextStyle(color: Colors.white)),
                      const SizedBox(height: 6),
                      Text('Est: $_totalCreditsNeeded cr • ₹$_totalPriceNeeded', style: TextStyle(color: Colors.white70, fontSize: 12)),
                    ],
                  ),
                ),
              )
            ],
          ),
        ),
      ),
    );
  }

  Widget _toolsToolbar() {
    // each tool shows selected state
    final tools = [
      {'key': 'trim', 'label': 'Trim'},
      {'key': 'filters', 'label': 'Filters'},
      {'key': 'voice_replace', 'label': 'Voice'},
      {'key': 'music_replace', 'label': 'Music'},
      {'key': 'background_change', 'label': 'Background'},
      {'key': 'auto_enhance', 'label': 'Enhance'},
      {'key': 'subtitle_edit', 'label': 'Subtitles'},
      {'key': 'aspect_change', 'label': 'Aspect'},
      {'key': 're_render', 'label': 'Re-render'},
      {'key': 'ai_rebuild', 'label': 'AI Rebuild'},
    ];

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12.0, vertical: 8),
      child: Wrap(
        spacing: 8,
        children: tools.map((t) {
          final k = t['key'] as String;
          final selected = _selectedEdits.containsKey(k);
          return ChoiceChip(
            selected: selected,
            label: Text(t['label']!),
            onSelected: (_) async {
              // special handling: open tool-specific modal if needed
              if (k == 'trim') {
                // open trim modal (simple)
                final res = await _openTrimDialog();
                if (res == true) _toggleFeature(k);
              } else if (k == 'filters') {
                final res = await _openFiltersDialog();
                if (res != null) _toggleFeature(k, res);
              } else if (k == 're_render') {
                final levels = await _openReRenderDialog();
                if (levels != null) _setReRenderLevels(levels);
              } else {
                _toggleFeature(k);
              }
            },
            selectedColor: Colors.deepPurpleAccent,
            backgroundColor: Colors.white10,
            labelStyle: const TextStyle(color: Colors.white),
          );
        }).toList(),
      ),
    );
  }

  Future<bool?> _openTrimDialog() async {
    // Simple trim dialog (start-end seconds)
    int start = 0;
    int end = _videoDuration.inSeconds;
    return showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: Color(0xFF0B0B0D),
        title: const Text('Trim Video', style: TextStyle(color: Colors.white)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('Start (s): $start', style: TextStyle(color: Colors.white70)),
            Slider(value: start.toDouble(), min: 0, max: _videoDuration.inSeconds.toDouble(), onChanged: (v) { start = v.toInt(); }),
            Text('End (s): $end', style: TextStyle(color: Colors.white70)),
            Slider(value: end.toDouble(), min: 0, max: _videoDuration.inSeconds.toDouble(), onChanged: (v) { end = v.toInt(); }),
            const SizedBox(height: 8),
            Text('Trim: $start - $end s', style: TextStyle(color: Colors.white)),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.of(context).pop(false), child: Text('Cancel')),
          ElevatedButton(onPressed: () => Navigator.of(context).pop(true), child: Text('Apply')),
        ],
      ),
    );
  }

  Future<Map<String, dynamic>?> _openFiltersDialog() async {
    // choose a filter preset
    String selected = 'Cinematic';
    final presets = ['Cinematic', 'Vintage', 'Neon Glow', 'Vlog', 'Black & White'];
    return showDialog<Map<String, dynamic>>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: Color(0xFF0B0B0D),
        title: const Text('Choose Filter', style: TextStyle(color: Colors.white)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: presets.map((p) => RadioListTile<String>(
            value: p, groupValue: selected, onChanged: (v) { selected = v!; setState(() {}); }, title: Text(p, style: TextStyle(color: Colors.white)),
          )).toList(),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.of(context).pop(null), child: Text('Cancel')),
          ElevatedButton(onPressed: () => Navigator.of(context).pop({'preset': selected}), child: Text('Apply')),
        ],
      ),
    );
  }

  Future<int?> _openReRenderDialog() async {
    int levels = 1;
    return showDialog<int>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: Color(0xFF0B0B0D),
        title: Text('Re-render levels', style: TextStyle(color: Colors.white)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('Increase quality / passes: $levels', style: TextStyle(color: Colors.white70)),
            Slider(value: levels.toDouble(), min: 1, max: 5, divisions: 4, onChanged: (v) => levels = v.toInt()),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.of(context).pop(null), child: Text('Cancel')),
          ElevatedButton(onPressed: () => Navigator.of(context).pop(levels), child: Text('Apply')),
        ],
      ),
    );
  }

  Widget _bottomActionsBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12.0, vertical: 10),
      child: Row(
        children: [
          Expanded(
            child: ElevatedButton.icon(
              onPressed: _selectedEdits.isEmpty ? null : _previewChanges,
              icon: const Icon(Icons.visibility),
              label: const Text('Preview'),
              style: ElevatedButton.styleFrom(backgroundColor: Colors.white10),
            ),
          ),
          const SizedBox(width: 8),
          ElevatedButton.icon(
            onPressed: _selectedEdits.isEmpty || _isSubmitting ? null : _confirmAndSubmitJob,
            icon: _isSubmitting ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)) : const Icon(Icons.send),
            label: Text(_isSubmitting ? 'Submitting...' : 'Confirm (${_totalCreditsNeeded}cr)'),
            style: ElevatedButton.styleFrom(backgroundColor: Colors.deepPurpleAccent),
          )
        ],
      ),
    );
  }

  // small helper showing current pricing JSON (preview)
  Widget _pricingPreview() {
    final txt = JsonEncoder.withIndent('  ').convert(_pricing);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12.0, vertical: 6),
      child: Blur(
        blur: 6,
        borderRadius: BorderRadius.circular(12),
        child: Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(color: Colors.white.withOpacity(0.02), borderRadius: BorderRadius.circular(12)),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Current Pricing (admin)', style: TextStyle(color: Colors.white70, fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            Text(txt, style: TextStyle(color: Colors.white54, fontSize: 12, fontFamily: 'monospace')),
          ]),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(children: [
        _animatedBackground(),
        SafeArea(
          child: Column(
            children: [
              _topBar(),
              _videoPreviewCard(),
              _pricingPreview(),
              _toolsToolbar(),
              const SizedBox(height: 8),
              Expanded(child: SingleChildScrollView(child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Selected Edits', style: GoogleFonts.poppins(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700)),
                    const SizedBox(height: 8),
                    if (_selectedEdits.isEmpty)
                      Text('No edits selected. Tap tools to add changes.', style: TextStyle(color: Colors.white70))
                    else
                      ..._selectedEdits.entries.map((e) => ListTile(
                        tileColor: Colors.white10,
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                        title: Text(e.key, style: TextStyle(color: Colors.white)),
                        subtitle: Text(e.value is Map ? e.value.toString() : e.value.toString(), style: TextStyle(color: Colors.white70)),
                        trailing: IconButton(icon: Icon(Icons.delete), onPressed: () { setState(() { _selectedEdits.remove(e.key); _recalculateCost(); }); }),
                      )),
                    const SizedBox(height: 20),
                  ],
                ),
              ))),
              _bottomActionsBar(),
              const SizedBox(height: 8),
            ],
          ),
        )
      ]),
    );
  }
}
